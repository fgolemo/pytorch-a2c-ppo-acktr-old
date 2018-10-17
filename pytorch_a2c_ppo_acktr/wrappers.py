import cv2
import gym
from gym import spaces
import numpy as np


class DuckietownRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -10
        elif reward > 0:
            reward += 10
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
            vels = [0.3, +1.0]
        # Sharp right
        elif action == 1:
            vels = [0.3, -1.0]
        # Go forward
        elif action == 2:
            vels = [0.5, 0.0]
        # slight left
        elif action == 3:
            vels = [0.5, +1.0]
        # slight right
        elif action == 4:
            vels = [0.5, -1.0]

        else:
            assert False, "unknown action"
        return np.array(vels)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, color=False):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.color = color
        colors = 1
        if color:
            colors = 3
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, colors), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if not self.color:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            return frame[:, :, None]
        else:
            return frame

class Normalize(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.obs_lo = self.observation_space.low[0,0,0]
        self.obs_hi = self.observation_space.high[0,0,0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)