from utils.utils_profiling import *  # load before other local modules

import argparse
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import dgl
import numpy as np
import torch
import csv
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
    pred = pred.argmax(1)
    y = y.argmax(1)
    num_correct = (pred == y).sum().item()
    #print(f'num of correct:{num_correct}')

    return num_correct


def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, scheduler, FLAGS):
    model.train()
    loss_epoch = 0

    num_iters = len(dataloader)


    for i, (g,y) in enumerate(dataloader):

        g = g.to(FLAGS.device)

        # B, 1
        cls = y.to(FLAGS.device)



        # run model forward and compute loss
        # B, 15

        pred = model(g)


        loss = loss_fnc(pred, cls)
        loss_epoch += to_np(loss)

        # gradient accumulation
        acc_steps = 1
        loss = loss/acc_steps


        loss.backward()

        if torch.isnan(loss):
            import pdb
            pdb.set_trace()

        # backprop
        if (i+1) % acc_steps == 0 or i+1 == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

        # print to console
        if i % FLAGS.print_interval == 0:
            print(
                f"[{epoch}|{i}] loss: {loss_epoch/(i+1):.5f}")


        if FLAGS.profile and i == 10:
            sys.exit()

    scheduler.step()
    loss_epoch /= len(dataloader)

    return loss_epoch


def test_epoch(epoch, model, loss_fnc, dataloader, FLAGS, dT=None):
    model.eval()

    count = 0
    acc_epoch = {'acc': 0.0}
    num_corr = 0
    loss_epoch = 0.0
    cum  = 0

    with torch.no_grad():
        for i, (g,y) in enumerate(dataloader):

            g = g.to(FLAGS.device)

            cls = y.to(FLAGS.device)

            # run model forward and compute loss

            pred = model(g)


            loss_epoch += loss_fnc(pred, cls) / len(dataloader)
            pred, cls = pred, cls
            acc = get_acc(pred, cls)
            num_corr += acc
            count += y.shape[0]
            acc_epoch['acc'] = num_corr/count

            # eval linear baseline
            # Apply linear update to locations.
        print(f"...[{epoch}|test] loss: {loss_epoch:.5f}")

        print(f"Acc is {acc_epoch}\n")



    return loss_epoch,acc_epoch['acc']



class RandomRotation(object):
    def __init__(self):
        pass

    def __call__(self, x):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        return x @ Q


def collate(samples):
    graphs,y = map(list, zip(*samples))
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

    model = models.__dict__.get(FLAGS.model)(FLAGS.num_layers, FLAGS.num_channels, num_degrees=FLAGS.num_degrees,
                                             div=FLAGS.div, n_heads=FLAGS.head, si_m=FLAGS.simid, si_e=FLAGS.siend,
                                             x_ij=FLAGS.xij, kernel=FLAGS.kernel, num_random=FLAGS.num_random,
                                             out_dim=FLAGS.num_points*2, num_class=FLAGS.num_class,
                                             batch=FLAGS.batch_size, num_points=FLAGS.num_points)


    # Save path
    save_path = os.path.join(FLAGS.save_dir, FLAGS.name + '.pt')

    if FLAGS.restore is not None:
        model.load_state_dict(torch.load(FLAGS.restore))
    model.to(FLAGS.device)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
    #optimizer = optim.SGD(model.parameters(),lr = FLAGS.lr,momentum = 0.9)

    #scheduler = optim.lr_scheduler.StepLR(optimizer, 25000, gamma=0.9)
    if FLAGS.kernel:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(FLAGS.device)
    task_loss = criterion



    # Run training
    print('Begin training')

    name = f'{FLAGS.num_points}_{FLAGS.num_random}_batch{FLAGS.batch_size}_{FLAGS.siend}_QPartial_{FLAGS.num_layers + 1}_{FLAGS.num_channels//FLAGS.div}'
    print(name)
    for epoch in range(FLAGS.num_epochs):
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

        train_loss = train_epoch(epoch, model, task_loss, train_loader, optimizer, scheduler, FLAGS)
        #base = time.time()

        test_loss,test_acc = test_epoch(epoch, model, task_loss, test_loader, FLAGS)
        #epoch_time = time.time() - base
        #print(f"one epoch costs {epoch_time}")


        with open(f'{name}_train_loss.csv','a') as f:
            f.write(f'{train_loss}')
            f.write('\n')
        with open(f'{name}_test_loss.csv','a') as f:
            f.write(f'{test_loss}')
            f.write('\n')
        with open(f'{name}_test_acc.csv','a') as f:
            f.write(f'{test_acc}')
            f.write('\n')



if __name__ == '__main__':

    FLAGS, UNPARSED_ARGV = get_flags()
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    name = f'{FLAGS.num_points}_{FLAGS.num_random}_batch16_si{FLAGS.siend}_Qpartial_{FLAGS.num_channels//FLAGS.div}'

    # Where the magic is
    try:
        main(FLAGS, UNPARSED_ARGV)
    except Exception:
        import pdb, traceback

        traceback.print_exc()
        pdb.post_mortem()
