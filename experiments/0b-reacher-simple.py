import argparse
import gym
import pybulletgym
import numpy as np
from itertools import count

import csv
import torch

from envs import *
from policies.simple import BobPolicy, AlicePolicy
import os
import psutil

parser = argparse.ArgumentParser(description='Initial Experiments for OGG')
parser.add_argument('--seed', type=int, default=-1, metavar='N',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='# of episodes to log in between')
parser.add_argument('--randomized-env-id', type=str, default='ReacherPyBulletEnv-v0')
parser.add_argument('--eval-env-id', type=str,default='ReacherPyBulletEnv-v0')

N_PROCS = 5
N_ROLLOUTS = 20

def evaluate_policy(env, policy, render=False):
    nepisodes = 0
    rewards = []
    
    while nepisodes < N_ROLLOUTS:
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            with torch.no_grad():
                action = policy.select_action(state, deterministic=True, save_log_probs=False)

            state, reward, done, _ = env.step(action)
            # env.render()
            ep_reward += reward           

        rewards.append(ep_reward)
        nepisodes += 1

    return rewards

def main(args):
    model_path = 'saved-models/expt-0b-mtncar/lowerlr-{}/{}'.format(args.randomized_env_id, args.seed)
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_env = gym.make(args.randomized_env_id)
    training_env.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    policy = BobPolicy(state_dim=2, action_dim=2)

    total_timesteps = 1e6
    timesteps = 0
    nepisodes = 0
    best_reward = -np.inf

    learning_curve = []

    while timesteps < total_timesteps:
        state = training_env.reset()
        done = False

        while not done:
            action = policy.select_action(state)
            state, reward, done, _ = training_env.step(action)
            policy.log(reward)
            
            timesteps += 1

        nepisodes += 1
        policy.finish_episode(gamma=0.99)

        if nepisodes % args.log_interval == 0:
            eval_rewards = evaluate_policy(training_env, policy, render= nepisodes % 50==0)
            eval_reward = np.mean(eval_rewards)
            learning_curve.append(eval_reward)

            torch.save(policy.policy.state_dict(), 
                os.path.join('{}/{}.pth'.format(model_path, nepisodes)))

            if eval_reward > best_reward: 
                best_reward = eval_reward                

                torch.save(policy.policy.state_dict(), 
                    os.path.join('{}/best.pth'.format(model_path)))

            print('[Seed: {}] Current: {}, Best: {}, Eps: {}, Timesteps: {}'.format(args.seed, eval_reward, best_reward, nepisodes, timesteps)) 

    np.savez(os.path.join('{}/learningcurve.npz'.format(model_path)), learning_curve=learning_curve)

if __name__ == '__main__':
    args = parser.parse_args()
    
    for seed in [100, 101, 102]:
        args.seed = seed
        main(args)