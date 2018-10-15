import gym
from gym import spaces
import numpy as np


class DuckietownRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -1
        else:
            reward += 4

        return reward


class DuckietownDiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (sharp left, sharp right, forward, sight left, slight right)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(5)

    def action(self, action):
        # Sharp left
        if action == 0:
            vels = [0.1, +1.0]
        # Sharp right
        elif action == 1:
            vels = [0.1, -1.0]
        # Go forward
        elif action == 2:
            vels = [0.7, 0.0]
        # slight left
        elif action == 3:
            vels = [0.5, +1.0]
        # slight right
        elif action == 4:
            vels = [0.5, -1.0]

        else:
            assert False, "unknown action"
        return np.array(vels)
