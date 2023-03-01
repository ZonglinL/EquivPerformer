import argparse
import torch
import numpy as np


def get_flags():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--model', type=str, default='SE3Transformer',
                        help="String name of model")
    parser.add_argument('--num_layers', type=int, default=4,
                        help="Number of equivariant layers")
    parser.add_argument('--num_degrees', type=int, default=4,
                        help="Number of irreps {0,1,...,num_degrees-1}")
    parser.add_argument('--num_channels', type=int, default=4,
                        help="Number of channels in middle layers")
    parser.add_argument('--div', type=float, default=1,
                        help="Low dimensional embedding fraction")
    parser.add_argument('--head', type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument('--kernel', action='store_true',
                        help="Performer or not")
    parser.add_argument('--pool_all', action='store_true',
                        help="Use three pools or not")


    # Type of self-interaction in attention layers,
    # valid: '1x1' (simple) and 'att' (attentive) with a lot more parameters
    parser.add_argument('--simid', type=str, default='1x1',)
    parser.add_argument('--siend', type=str, default='att')
    parser.add_argument('--xij', type=str, default='add')

    # Meta-parameters
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=5e-3,
                        help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=50000,
                        help="Number of epochs")
    parser.add_argument('--num_random', type=int, default=20,
                        help="Number of random features")
    parser.add_argument('--num_points', type=int, default=320,
                        help="Number of points to keep")
    parser.add_argument('--antithetic', action='store_true',
                        help="whether to use antithetic sampling")
    # Data
    parser.add_argument('--num_class', type=int, default=15,
                        help="Number of Classes of data objects")
    # location of data for relational inference
    parser.add_argument('--data', type=str, default='experiments/pc3d/data/')
    parser.add_argument('--data_str', type=str, default='no_bg')


    # Logging
    parser.add_argument('--name', type=str, default='pc3d_dgl', help="Run name")
    parser.add_argument('--log_interval', type=int, default=25,
                        help="Number of steps between logging key stats")
    parser.add_argument('--print_interval', type=int, default=100,
                        help="Number of steps between printing key stats")
    parser.add_argument('--save_dir', type=str, default="models",
                        help="Directory name to save models")
    parser.add_argument('--restore', type=str, default=None,
                        help="Path to model to restore")
    parser.add_argument('--verbose', type=int, default=0)

    # Miscellanea
    parser.add_argument('--num_workers', type=int, default=2,
                        help="Number of data loader workers")
    parser.add_argument('--profile', action='store_true',
                        help="Exit after 10 steps for profiling")

    # Random seed for both Numpy and Pytorch
    parser.add_argument('--seed', type=int, default=1)

    FLAGS, UNPARSED_ARGV = parser.parse_known_args()


    # Automatically choose GPU if available
    if torch.cuda.is_available():
        FLAGS.device = torch.device('cuda:0')
    else:
        FLAGS.device = torch.device('cpu')

    print("\n\nFLAGS:", FLAGS)
    print("UNPARSED_ARGV:", UNPARSED_ARGV, "\n\n")

    return FLAGS, UNPARSED_ARGV
