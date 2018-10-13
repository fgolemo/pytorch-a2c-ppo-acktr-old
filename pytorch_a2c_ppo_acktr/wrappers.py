import gym


class DuckietownRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -1
        else:
            reward += 4

        return reward

