import numpy as np
import torch
from torch.autograd import Variable

class Inference(object):
    def __init__(self, model, seed=1, num_stack=1, normalized=1):
        self.model = model
        self.seed = seed
        self.num_stack = num_stack
        self.normalized = normalized
        self.actor_critic, _ = torch.load(self.model)
        print (self.actor_critic)
        # print (self.ob_rms)

        self.states = torch.zeros(1, self.actor_critic.state_size)
        self.masks = torch.zeros(1, 1)

    def get_action(self, obs):
        obs = torch.from_numpy(obs).unsqueeze(0)
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
                    "ErgoFightStatic-Headless-Fencing-v0-180209225957.pt")

    env = gym.make("ErgoFightStatic-Graphical-Fencing-v0")
    obs = env.reset()

    for i in range(20):
        action = inf.get_action(obs)
        print (obs)
        print (action)
        print ("---")
        obs, rew, _, _ = env.step(action)

