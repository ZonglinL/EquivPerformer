from utils.utils_profiling import *  # load before other local modules

import argparse
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import numpy as np
import torch
import wandb
import time
import datetime

from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from experiments.pc3d.pccls_dataloader import PC3DDataset
from utils import utils_logging

from experiments.pc3d import pccls_models as models
from equivariant_attention.from_se3cnn.SO3 import rot
from experiments.pc3d.pccls_flags import get_flags


def to_np(x):
    return x.cpu().detach().numpy()


def get_acc(pred, y, verbose=True):

    pred = pred.argmax(-1)
    y = y.argmax(-1)
    num_correct = (pred==y).sum().item()

    return num_correct


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS):
    model.train()
    loss_epoch = 0

    num_iters = len(dataloader)

    wandb.log({"lr": optimizer.param_groups[0]['lr']}, commit=False)
    lag_step = FLAGS.lag_step
    for i, (g, y) in enumerate(dataloader):

        g = g.to(FLAGS.device)
        # B, 1
        cls = y.to(FLAGS.device)

        # run model forward and compute loss
        # B, 15
        pred = model(g)
        loss = loss_fnc(pred, cls)
        loss_epoch += to_np(loss)

        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

        # backprop
        loss /= lag_step
        loss.backward()
        if (i+1)%lag_step == 0 or i+1 == num_iters:
            optimizer.step()
            optimizer.zero_grad()


        # print to console
        if i % FLAGS.print_interval == 0:
            print(
                f"[{epoch}|{i}] loss: {loss_epoch/(i+1):.5f}")

        # log to wandb

        if i % FLAGS.log_interval == 0:
            # 'commit' is only set to True here, meaning that this is where
            # wandb counts the steps
            wandb.log({"Train Batch Loss": loss_epoch/(i+1)}, commit=True)
            
        # exit early if only do profiling
        if FLAGS.profile and i == 10:
            sys.exit()

    scheduler.step()

    # log train accuracy for entire epoch to wandb
    loss_epoch /= num_iters
    wandb.log({"Train Epoch Loss": loss_epoch}, commit=False)


def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS, dT= None):
    model.eval()


    #acc_epoch = {'acc': 0.0 }
    #acc_epoch_blc = {'acc': 0.0}  # for constant baseline
    #acc_epoch_bll = {'acc': 0.0}  # for linear baseline
    loss_epoch = 0.0
    num_data = 0
    num_correct = 0
    for i, (g, y) in enumerate(dataloader):
        g = g.to(FLAGS.device)
        cls = y.to(FLAGS.device)

        # run model forward and compute loss
        pred = model(g).detach()

        loss_epoch += to_np(loss_fnc(pred, cls))*pred.shape[0]
        pred, cls = to_np(pred), to_np(cls)

        num_correct += get_acc(pred,cls)
        num_data += pred.shape[0]


        #print(num_correct)

    #print(num_correct)
    acc_epoch = num_correct/num_data
    loss_epoch /= num_data

    print(f"...[{epoch}|test] loss: {loss_epoch:.5f}")

    print(f"Acc is {acc_epoch}\n")


    wandb.log({"Test loss": loss_epoch}, commit=False)
    wandb.log({"Test Acc": acc_epoch}, commit=False)



class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.stack(y)


def main(FLAGS, UNPARSED_ARGV):
    # Prepare data
    train_dataset = PC3DDataset(FLAGS, split='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size,
                              shuffle=True,
                              collate_fn=collate,
                              num_workers=FLAGS.num_workers,
                              drop_last=True)

    test_dataset = PC3DDataset(FLAGS, split='test')
    # drop_last is only here so that we can count accuracy correctly;
    test_loader = DataLoader(test_dataset,
                             batch_size=FLAGS.batch_size,
                             shuffle=False,
                             collate_fn=collate,
                             num_workers=FLAGS.num_workers,
                             drop_last=True)

    # time ste


    FLAGS.train_size = len(train_dataset)
    FLAGS.test_size = len(test_dataset)
    assert len(test_dataset) < len(train_dataset)

    model = models.__dict__.get(FLAGS.model)(FLAGS.num_layers, FLAGS.num_channels, num_degrees=FLAGS.num_degrees,
                                             div=FLAGS.div, n_heads=FLAGS.head, si_m=FLAGS.simid, si_e=FLAGS.siend,
                                             x_ij=FLAGS.xij, kernel=FLAGS.kernel, num_random=FLAGS.num_random,
                                             out_dim=FLAGS.batch_size*FLAGS.num_points, num_class=FLAGS.num_class,
                                             batch=FLAGS.batch_size, antithetic=FLAGS.antithetic,num_points = FLAGS.num_points)

    #utils_logging.write_info_file(model, FLAGS=FLAGS, UNPARSED_ARGV=UNPARSED_ARGV, wandb_log_dir=wandb.run.dir)

    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr,weight_decay=FLAGS.weight_decay)
    #optimizer = optim.SGD(model.parameters(), momentum= 0.9, lr=FLAGS.lr/10)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, FLAGS.num_epochs, eta_min=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.8)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion

    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')

    # Run training
    print('Begin training')
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS)
        test_epoch(epoch, model, task_loss, test_loader, FLAGS)


if __name__ == '__main__':

    FLAGS, UNPARSED_ARGV = get_flags()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    # Log all args to wandb

    wandb.init(project='equivariant-attention-pccls', name=FLAGS.name, config=FLAGS)
    wandb.save('*.txt')

    # Where the magic is
    try:
        main(FLAGS, UNPARSED_ARGV)
    except Exception:
        import pdb, traceback
        traceback.print_exc()
        pdb.post_mortem()
