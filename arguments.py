import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='DQN', help='algorithm to use: DQN | A2C | PPO')
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=20000,
        help='number of environment steps to train (default: 10e5)')
    
    parser.add_argument(
        '--env-name',
        default='golddigger',
        help='environment to train on (default: golddigger)')
    parser.add_argument(
        '--use-one-hot',
        action='store_true',
        default=False,
        help='use one hot representation')

    parser.add_argument(
        '--use-deterministic',
        action='store_true',
        default=False,
        help='deterministic policy')

    parser.add_argument(
        '--use-local-observation',
        action='store_true',
        default=False,
        help='use local observation')
    parser.add_argument(
        '--gpuid',
        default=0,
        help="gpu")
    parser.add_argument(
        '--log-dir',
        default='./exp/log/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./exp/trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args

