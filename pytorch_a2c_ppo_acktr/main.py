import glob
import copy
import os
import time

import numpy as np
import torch
from shutil import copyfile

# Hyperdash is convenient experiment monitor for your phone
from pytorch_a2c_ppo_acktr import algo

has_hyperdash = False
try:
    from hyperdash import Experiment

    has_hyperdash = True
except ImportError:
    # if we don't have Hyperdash, no problem
    pass

from pytorch_a2c_ppo_acktr.arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from pytorch_a2c_ppo_acktr.envs import make_env
from pytorch_a2c_ppo_acktr.model import Policy
from pytorch_a2c_ppo_acktr.storage import RolloutStorage
from pytorch_a2c_ppo_acktr.visualize import visdom_plot
from pytorch_a2c_ppo_acktr.utils import update_current_obs

args = get_args()

if args.memdebug:
    import gc
    import objgraph
    import ipdb

run_name = time.strftime("%y%m%d%H%M%S")

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

exp = None
if has_hyperdash:
    exp = Experiment("{} - {}".format(args.env_name, args.algo))
    exp.param("NAME", run_name)
    for param, value in vars(args).items():
        exp.param(param, value)

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

num_breaks = num_updates / 10

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    print("#######")
    print(
        "WARNING: All rewards are clipped or "
        "so you need to use a monitor (see envs.py) or "
        "visdom plot to get true rewards")
    print("#######")

    torch.set_num_threads(1)

    highest_mean_reward = -9999999

    if args.vis:
        print("===USING VISDOM===")
        from visdom import Visdom
        viz = Visdom(server=args.vis_host, port=args.vis_port, ipv6=False)
        win = None

    envs = [make_env(args.env_name, args.seed, i, args.log_dir, args.add_timestep, args.custom_gym, args.scale_img, args.duckietown)
            for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=args.gamma)

    obs_shape = envs.observation_space.shape
    print("original obs shape: {}, {}".format(obs_shape, args.num_stack))
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
    print("final obs shape: {}".format(obs_shape))

    actor_critic = Policy(obs_shape, envs.action_space, args.recurrent_policy, args.normalized)

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                         eps=args.eps,
                         max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    obs = envs.reset()
    update_current_obs(obs, current_obs, obs_shape, args.num_stack)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs, current_obs, obs_shape, args.num_stack)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            save_model_path = os.path.join(
                save_path,
                args.env_name + "-" + run_name)
            torch.save(save_model, save_model_path + ".pt")

            if final_rewards.mean() > highest_mean_reward:
                highest_mean_reward = final_rewards.mean()

                copyfile(save_model_path + ".pt", save_model_path + "-best.pt")

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                    format(j, total_num_steps,
                           int(total_num_steps / (end - start)),
                           final_rewards.mean(),
                           final_rewards.median(),
                           final_rewards.min(),
                           final_rewards.max(), dist_entropy,
                           value_loss, action_loss))
            if exp is not None:
                exp.metric("mean reward", float(final_rewards.mean().numpy()))
                exp.metric("min reward", float(final_rewards.min().numpy()))
                exp.metric("max reward", float(final_rewards.max().numpy()))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                print("Got an IO error when plotting vizdom")
                pass

        if args.robot and j % num_breaks == 0:
            envs.venv.reset()
            input("\n[ROBOT MAINTENANCE]. Press Enter to continue...")

        if args.memdebug and j % 5 == 0:
            gc.collect()  # don't care about stuff that would be garbage collected properly
            print(objgraph.show_most_common_types())
            ipdb.set_trace()

    if has_hyperdash:
        exp.end()


if __name__ == "__main__":
    main()
