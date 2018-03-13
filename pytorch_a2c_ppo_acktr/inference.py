import numpy as np
import torch
from torch.autograd import Variable
import sys
import pytorch_a2c_ppo_acktr.model as pt_model
import pytorch_a2c_ppo_acktr.utils as pt_utils
import pytorch_a2c_ppo_acktr.distributions as pt_dist


class Inference(object):
    def __init__(self, model, seed=1, num_stack=1, normalized=1):
        self.model = model
        self.seed = seed
        self.num_stack = num_stack
        self.normalized = normalized

        # constants
        self.clipob = 10.0
        self.cliprew = 10.0
        self.epsilon = 1e-8

        print("sys.modules", sys.modules)
        sys.modules['model'] = pt_model
        sys.modules['distributions'] = pt_dist
        sys.modules['utils'] = pt_utils
        self.actor_critic, self.ob_rms = torch.load(self.model)

        self.states = torch.zeros(1, self.actor_critic.state_size)
        self.masks = torch.zeros(1, 1)

        # self.obs_buffer1 = []
        # self.obs_buffer2 = []

    def normalize_obs(self, obs):
        self.ob_rms.update(obs)
        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        return obs

    def get_action(self, obs):
        # self.obs_buffer1.append(obs) # this was only for debugging
        obs = self.normalize_obs(obs)
        # self.obs_buffer2.append(obs) # only for debugging

        obs = torch.from_numpy(obs).float().unsqueeze(0)

        value, action, _, states = self.actor_critic.act(Variable(obs, volatile=True),
                                                         Variable(self.states, volatile=True),
                                                         Variable(self.masks, volatile=True),
                                                         deterministic=True)
        self.states = states.data
        cpu_actions = action.data.squeeze(1).cpu().numpy()[0]

        return cpu_actions


if __name__ == '__main__':
    import gym
    import gym_vrep

    inf = Inference("/home/florian/dev/pytorch-a2c-ppo-acktr/"
                    "trained_models/ppo/"
                    "ErgoFightStatic-Headless-Fencing-v0-180301140937.pt")

    env = gym.make("ErgoFightStatic-Graphical-Fencing-v0")
    obs = env.reset()

    import matplotlib.pyplot as plt

    att_actions = []
    for i in range(100):
        action = inf.get_action(obs)
        print (obs)
        print (action)
        print (" ")
        att_actions.append(action)
        # print (obs)
        # print (action)
        # print ("---")
        obs, rew, _, _ = env.step(action)

    # att_actions = np.array(att_actions)
    # print(att_actions.shape)
    #
    # for i in range(6):
    #     plt.plot(range(len(att_actions)), att_actions[:, i])

    obs_buffer = np.array(inf.obs_buffer2)

    f, axarr = plt.subplots(2, 2)

    plot_pos = 0
    plot_vel = 0
    title = "robot 0 pos"
    for i in range(24):
        if i == 6:
            plot_vel = 1
            title = "robot 0 vel"
        if i == 12:
            plot_vel = 0
            plot_pos = 1
            title = "robot 1 pos"
        if i == 18:
            plot_vel = 1
            title = "robot 1 vel"

        axarr[plot_vel, plot_pos].plot(range(len(obs_buffer)), obs_buffer[:, i], label=int(i-(np.floor(i/6)*6)))
        axarr[plot_vel, plot_pos].set_title(title)
        axarr[plot_vel, plot_pos].legend()

    plt.show()
