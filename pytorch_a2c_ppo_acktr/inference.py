import argparse
import types

import numpy as np
import torch
from torch.autograd import Variable
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from pytorch_a2c_ppo_acktr.envs import make_env

class Inference(object):
    def __init__(self, model, seed=1, num_stack=1, normalized=1):
        self.model = model
        self.seed = seed
        self.num_stack = num_stack
        self.normalized = normalized
        self.actor_critic, self.ob_rms = torch.load(self.model)
        print (self.actor_critic)
        print (self.ob_rms)

# if len(env.observation_space.shape) == 1:
#     env = VecNormalize(env, ret=False)
#     env.ob_rms = ob_rms
#
#
#     # An ugly hack to remove updates
#     def _obfilt(self, obs):
#         if self.ob_rms:
#             obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
#             return obs
#         else:
#             return obs
#
#
#     env._obfilt = types.MethodType(_obfilt, env)
#     render_func = env.venv.envs[0].render
# else:
#     render_func = env.envs[0].render
#
# obs_shape = env.observation_space.shape
# print("obs shape:", obs_shape)
# obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
# print("obs shape:", obs_shape)
# current_obs = torch.zeros(1, *obs_shape)
# states = torch.zeros(1, actor_critic.state_size)
# masks = torch.zeros(1, 1)
#
#
# def update_current_obs(obs):
#     shape_dim0 = env.observation_space.shape[0]
#     obs = torch.from_numpy(obs).float()
#     if args.num_stack > 1:
#         current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
#     current_obs[:, -shape_dim0:] = obs
#
#
# render_func('human')
# obs = env.reset()
# update_current_obs(obs)
#
# if args.env_name.find('Bullet') > -1:
#     import pybullet as p
#
#     torsoId = -1
#     for i in range(p.getNumBodies()):
#         if (p.getBodyInfo(i)[0].decode() == "torso"):
#             torsoId = i
#
# while True:
#     value, action, _, states = actor_critic.act(Variable(current_obs, volatile=True),
#                                                 Variable(states, volatile=True),
#                                                 Variable(masks, volatile=True),
#                                                 deterministic=True)
#     states = states.data
#     cpu_actions = action.data.squeeze(1).cpu().numpy()
#     # Obser reward and next obs
#     obs, reward, done, _ = env.step(cpu_actions)
#
#     masks.fill_(0.0 if done else 1.0)
#
#     if current_obs.dim() == 4:
#         current_obs *= masks.unsqueeze(2).unsqueeze(2)
#     else:
#         current_obs *= masks
#     update_current_obs(obs)
#
#     if args.env_name.find('Bullet') > -1:
#         if torsoId > -1:
#             distance = 5
#             yaw = 0
#             humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
#             p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)
#
#     render_func('human')
#     # time.sleep(.05)

if __name__ == '__main__':
    inf = Inference("/home/florian/dev/pytorch-a2c-ppo-acktr/"
                    "trained_models/ppo/"
                    "ErgoFightStatic-Headless-Fencing-v0-180209225957.pt")
