import os

import gym
import numpy as np
from gym.spaces.box import Box
import importlib

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

from pytorch_a2c_ppo_acktr.wrappers import DuckietownRewardWrapper, DuckietownDiscreteWrapper, WarpFrame, Normalize

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, add_timestep,
             custom_gym=None, scale_img=False, duckietown=False, dt_discrete=False, color_img=False):
    def _thunk():
        print("CUSTOM GYM:", custom_gym)
        if custom_gym is not None and custom_gym != "":
            module = importlib.import_module(custom_gym, package=None)
            print("imported env '{}'".format((custom_gym)))

        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        if not is_atari and scale_img:
            env = WarpFrame(env, color_img)

        if duckietown:
            env = DuckietownRewardWrapper(env)
            if dt_discrete:
                env = DuckietownDiscreteWrapper(env)

        env = Normalize(env)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape
        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        if is_atari:
            env = wrap_deepmind(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)

        return env

    return _thunk


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
