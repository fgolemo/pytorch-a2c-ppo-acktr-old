import argparse

import time
import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='a2c',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 1)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--num-stack', type=int, default=1,
                        help='number of frames to stack (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--vis-interval', type=int, default=100,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-frames', type=int, default=10e6,
                        help='number of frames to train (default: 10e6)')
    parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='/tmp/gym',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='../trained_models/',
                        help='directory to save agent logs (default: ../trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--scale-img', action='store_true', default=False,
                        help='make img obs into 84x84')
    parser.add_argument('--duckietown', action='store_true', default=False,
                        help='add duckietown-specific wrappers')
    parser.add_argument('--dt-discrete', action='store_true', default=False,
                        help='make duckietown env discrete')
    parser.add_argument('--color-img', action='store_true', default=False,
                        help='if false, the image is grayscale')
    parser.add_argument('--max-ep', type=int, default=500,
                        help="maximum steps of the environment if it doesn't have any (default: 500)")
    parser.add_argument('--cliprew', type=int, default=10,
                        help='reward clipping (default: 10)')
    parser.add_argument('--no-vis', action='store_true', default=False,
                        help='disables visdom visualization')
    parser.add_argument('--vis-host', type=str, default="http://localhost",
                        help='host for visdom, usually localhost, but can be remote')
    parser.add_argument('--vis-port', type=int, default=8097,
                        help='port for visdom, usually 80, but ')
    parser.add_argument('--custom-gym', type=str, default="",
                        help='for if you need to import a custom gym module')
    parser.add_argument('--normalized', action='store_true', default=False,
                        help='is the action space normalized? Means actions will be in [-1,1]')
    parser.add_argument('--robot', action='store_true', default=False,
                        help='for robot maintenance - pauses the experiment every num-frames/10 steps')
    parser.add_argument('--memdebug', '-mdbg', action='store_true', default=False,
                        help='pause every 20 iterations to enable memory profiling')


    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis
    args.log_dir = args.log_dir + time.strftime("%y%m%d-%H%M%S%f")

    return args
