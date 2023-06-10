import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import os
import time

import sumo_env_safe

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="sumo",
                    help='Gym environment (default: sumo)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=2, metavar='N',
                    help='random seed (default: 2)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--save-id', default="1",
                    help='Checkpoint ID (default: 1)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
# env = gym.make(args.env_name)
# env = sumo_env_safe.SumoEnvSafe(gui_f=False, config="road_network/data4/quickstart.sumocfg", netPath="road_network/data4/quickstart.net.xml", speed_mode=0)
# env = sumo_env_safe.SumoEnvSafe(gui_f=False, config="road_network/data3/quickstart.sumocfg", netPath="road_network/data3/quickstart.net.xml", speed_mode=0,route="")
env = sumo_env_safe.SumoEnvSafe(gui_f=False, config="road_network/data7/quickstart.sumocfg", netPath="road_network/data7/test_1km.net.xml", speed_mode=0, time_factor=0.1, disregardTime=False, route="route_2", terminationType="drivingDistance", targetDistance=3000)

env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else "", args.save_id))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

history = {
        'training_steps': [],
        'steps_per_s': [],
        'eval_return': [],
        'rewards': {
            'train': [],
            'test': []
        },
        'losses': {
            'critic_1_loss': [],
            'critic_2_loss': [],
            'entropy_loss': [],
            'policy_loss': [],
            'alpha_loss': []
        }
    }

save_path = f'checkpoints/{args.env_name}/id_{args.save_id}'
if not os.path.exists(save_path):
        os.makedirs(save_path)

print(f'{"Episode":>11}|{"Total steps":>11}|{"Episode steps":>11}|' +
          f'{"Reward":>11}|{"Entropy loss":>11}' + '\n' + 59 * '_',
          file=open(os.path.join(save_path, f'seed{args.seed}.out'), 'w'))

def save_history(env_name, suffix="", seed = 2,dir_path=None):
    if dir_path is None:
        dir_path = f'checkpoints/{env_name}/id_{suffix}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # if ckpt_path is None:
    #     ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
    ckpt_path = f'{dir_path}/seed{seed}.npy'
    print('Saving history to {}'.format(ckpt_path))
    np.save(ckpt_path, history)

# Training Loop
total_numsteps = 0
updates = 0

with torch.autograd.set_detect_anomaly(True):
    for i_episode in itertools.count(1):
        episode_reward = 0
        ep_rewards = []
        episode_steps = 0
        done = False
        state = env.reset()
        t0 = time.time()
        tot_ep_speed = 0
        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)

                    history['losses']['critic_1_loss'].append(critic_1_loss)
                    history['losses']['critic_2_loss'].append(critic_2_loss)
                    history['losses']['policy_loss'].append(policy_loss)
                    history['losses']['entropy_loss'].append(ent_loss)
                    history['losses']['alpha_loss'].append(alpha)
                    updates += 1

            next_state, reward, done, info = env.step(action) # Step
            print(f'Reward: {reward}, List: {env.get_reward()}')
            print(f'Next state: {next_state}')
            print(f'Action: {action}')
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            ep_rewards.append(reward)

            tot_ep_speed += info['velocity']

            isCollided = info['is_collided']
            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            if isCollided:
                mask = 1

            action_by_agent = info['action']
            print(f'Action from agent: {action_by_agent}')
            print(f'Action shape: {np.shape(action)}')
            print(f'Action by agent shape: {np.shape(action_by_agent)}')

            # break
            memory.push(state, action_by_agent, reward, next_state, mask) # Append transition to memory

            state = next_state

        if total_numsteps > args.num_steps:
            break

        writer.add_scalar('reward/train', episode_reward, i_episode)

        avg_speed = tot_ep_speed / episode_steps
        writer.add_scalar('train/avg_speed_per_episode', avg_speed, i_episode)
        writer.add_scalar('train/mean_reward_per_episode', np.mean(ep_rewards), i_episode)
        writer.add_scalar('train/collision', float(isCollided), i_episode)
        writer.add_scalar('train/episode_steps', episode_steps, i_episode)
        history['rewards']['train'].append(episode_reward)

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
        
        if i_episode % 10 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                ep_len = 0
                done = False
                while not (done or ep_len == env._max_episode_steps):
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward
                    ep_len += 1

                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

            writer.add_scalar('avg_reward/test', avg_reward, i_episode)
            history['rewards']['test'].append(avg_reward)
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")
        
            print(f'{i_episode:>11}',
                f'{total_numsteps:>11}',
                f'{episode_steps:>11}',
                f'{avg_reward:>11.2f}',
                f'{ent_loss:>11.5f}',
                file=open(os.path.join(save_path, f'seed{args.seed}.out'), 'a'))
        # save checkpoint and history
        agent.save_checkpoint(env_name=args.env_name, seed=args.seed, suffix=args.save_id, dir_path=save_path)
        save_history(env_name=args.env_name, seed=args.seed, suffix=args.save_id, dir_path=save_path)

env.close()

