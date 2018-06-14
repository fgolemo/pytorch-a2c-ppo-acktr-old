import argparse
import time
import types

import numpy as np
import torch
from torch.autograd import Variable
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from pytorch_a2c_ppo_acktr.envs import make_env

parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--log-dir', default='/tmp/gym/',
                    help='directory to save agent logs (default: /tmp/gym)')
parser.add_argument('--custom-gym', default='',
                    help='if you need to import a python package to load this gym environment, this is the place')
parser.add_argument('--model', default='',
                    help='include the path to the trained model file')
parser.add_argument('--normalized', type=int, default=1,
                    help='is the action space normalized? 1 for yes, 0 for no. 1 means actions will be in [0,1]')
parser.add_argument('--episodes', '-ep', type=int, default=0,
                    help='run for how many episodes?, set to 0 for unlimited')
parser.add_argument('--gather-rewards', '-gr', action='store_true', default=False,
                    help='save epdisode rewards to a file')

args = parser.parse_args()

if not args.gather_rewards:
    print ("===REWARDS ARE NOT BEING RECORDED===")


env = make_env(args.env_name, args.seed, 0, None, custom_gym=args.custom_gym)
env = DummyVecEnv([env])

actor_critic, ob_rms = \
    torch.load(args.model)

if len(env.observation_space.shape) == 1:
    env = VecNormalize(env, ret=False)
    env.ob_rms = ob_rms


    # An ugly hack to remove updates
    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs


    env._obfilt = types.MethodType(_obfilt, env)
    render_func = env.venv.envs[0].render
else:
    render_func = env.envs[0].render

obs_shape = env.observation_space.shape
print("obs shape:", obs_shape)
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
print("obs shape:", obs_shape)
current_obs = torch.zeros(1, *obs_shape)
states = torch.zeros(1, actor_critic.state_size)
masks = torch.zeros(1, 1)


def update_current_obs(obs):
    shape_dim0 = env.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs


render_func('human')
obs = env.reset()
update_current_obs(obs)

if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i


def write_rewards(rewards):
    filename = "{}-rewards-{}-{}.npz".format(
        args.model[:-3],
        args.episodes,
        time.strftime("%y%m%d%H%M%S"))
    np.savez(filename, rewards=rewards)
    print("wrote results to file:", filename)


episode = 0
rewards = []
reward_buf = 0
while True:
    value, action, _, states = actor_critic.act(Variable(current_obs, volatile=True),
                                                Variable(states, volatile=True),
                                                Variable(masks, volatile=True),
                                                deterministic=True)
    states = states.data
    cpu_actions = action.data.squeeze(1).cpu().numpy()
    # Obser reward and next obs
    obs, reward, done, _ = env.step(cpu_actions)
    reward_buf += reward

    masks.fill_(0.0 if done else 1.0)

    if current_obs.dim() == 4:
        current_obs *= masks.unsqueeze(2).unsqueeze(2)
    else:
        current_obs *= masks
    update_current_obs(obs)

    if args.env_name.find('Bullet') > -1:
        if torsoId > -1:
            distance = 5
            yaw = 0
            humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
            p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

    render_func('human')
    time.sleep(.01)

    if done:
        rewards.append(reward_buf)
        reward_buf = 0
        env.reset()

    if done and args.episodes != 0:
        episode += 1
        if episode == args.episodes:
            if args.gather_rewards:
                write_rewards(rewards)
            break
