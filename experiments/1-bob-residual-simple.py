import argparse
import gym
import numpy as np
from itertools import count

import csv
import torch

from envs import *
from envs.wrappers import RandomizedEnvWrapper
from envs.randomized_vecenv import make_vec_envs
from policies.simple import BobPolicy, AlicePolicy
from envs.heuristics.lunar.heuristics import heuristic, uncalibrated

import os
import psutil

parser = argparse.ArgumentParser(description='Initial Experiments for OGG')
parser.add_argument('--seed', type=int, default=-1, metavar='N',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='# of episodes to log in between')
parser.add_argument('--randomized-env-id', type=str, default='LunarLanderDefault-v0')
parser.add_argument('--eval-env-id', type=str,default='LunarLanderDefault-v0')

N_PROCS = 5
N_ROLLOUTS = 20

def evaluate_policy(env, policy):
    nepisodes = 0
    rewards = []
    state = env.reset()
    ep_reward = np.zeros(N_PROCS)

    while nepisodes < N_ROLLOUTS:        
        with torch.no_grad():
            uncalibrated_action = np.array([uncalibrated(env, state[i]) for i in range(N_PROCS)])
            residual_action = policy.select_action(state, deterministic=False, save_log_probs=False)[0]
            action = uncalibrated_action + residual_action

        state, reward, done, _ = env.step(action)
        
        for i, d in enumerate(done):
            if d:
                rewards.append(ep_reward[i])
                ep_reward[i] = 0
                nepisodes += 1
            else:
                ep_reward[i] += reward[i]

    return rewards

def main():
    args = parser.parse_args()

    model_path = 'saved-models/expt-1/{}/{}'.format(args.randomized_env_id, args.seed)
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_env = gym.make(args.randomized_env_id)

    if args.eval_env_id is None:
        args.eval_env_id = args.randomized_env_id

    rollout_env = make_vec_envs(args.eval_env_id, args.seed, N_PROCS)

    torch.manual_seed(args.seed)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    policy = BobPolicy()

    total_timesteps = 1e6
    timesteps = 0
    nepisodes = 0
    best_reward = -np.inf

    learning_curve = []

    while timesteps < total_timesteps:
        state = training_env.reset()
        done = False

        while not done:
            uncalibrated_action = uncalibrated(training_env, state) 
            residual_action = policy.select_action(state)

            action = uncalibrated_action + residual_action

            state, reward, done, _ = training_env.step(action)
            policy.log(reward)
            
            timesteps += 1

        nepisodes += 1
        policy.finish_episode(gamma=0.99)

        if nepisodes % args.log_interval == 0:
            eval_rewards = evaluate_policy(rollout_env, policy)
            eval_reward = np.mean(eval_rewards)
            learning_curve.append(eval_reward)

            torch.save(policy.policy.state_dict(), 
                os.path.join('{}/{}.pth'.format(model_path, nepisodes)))

            if eval_reward > best_reward: 
                best_reward = eval_reward                

                torch.save(policy.policy.state_dict(), 
                    os.path.join('{}/best.pth'.format(model_path)))

            print('Current: {}, Best: {}, Eps: {}, Timesteps: {}'.format(eval_reward, best_reward, nepisodes, timesteps)) 

    np.savez(os.path.join('{}/learningcurve.npz'.format(model_path)), learning_curve=learning_curve)

if __name__ == '__main__':
    main()