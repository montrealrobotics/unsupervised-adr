import argparse
import gym
import pybulletgym
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


class Arguments:
    randomized_env_id = 'ReacherPyBulletEnv-v0'
    eval_env_id = 'ReacherPyBulletEnv-v0'
    log_interval = 10

    def __init__(self, seed, sp_gamma, sp_percent):
        self.seed = seed
        self.sp_gamma = sp_gamma
        self.sp_percent = sp_percent
    

parser = argparse.ArgumentParser(description='Initial Experiments for OGG')
parser.add_argument('--seed', type=int, default=-1, metavar='N',
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='# of episodes to log in between')
parser.add_argument('--sp-gamma', type=float, default=0.1, metavar='N',
                    help='Self Play gamma')
parser.add_argument('--sp-percent', type=float, default=0.1, metavar='N',
                    help='Self Play percent')
parser.add_argument('--randomized-env-id', type=str, default='ReacherPyBulletEnv-v0')
parser.add_argument('--eval-env-id', type=str,default='ReacherPyBulletEnv-v0')

N_PROCS = 5
N_ROLLOUTS = 20
MAX_TIMESTEPS = 100
STATE_DIM = 8
TAU = 0.1


def check_closeness(state, goal):
    dist =  np.linalg.norm(state - goal) ** 2

    return dist < 0.1    

def evaluate_policy(env, policy):
    nepisodes = 0
    rewards = []
    state = env.reset()
    ep_reward = np.zeros(N_PROCS)

    full_state = np.zeros((N_PROCS, STATE_DIM * 2))
    full_state[:, :STATE_DIM] = state

    while nepisodes < N_ROLLOUTS:        
        with torch.no_grad():
            uncalibrated_action = np.array([uncalibrated(env, state[i]) for i in range(N_PROCS)])
            residual_action = policy.select_action(full_state, deterministic=True, save_log_probs=False)[0]
            action = uncalibrated_action + residual_action

        state, reward, done, _ = env.step(action)
        full_state[:, :STATE_DIM] = state

        for i, d in enumerate(done):
            if d:
                rewards.append(ep_reward[i])
                ep_reward[i] = 0
                nepisodes += 1
            else:
                ep_reward[i] += reward[i]

    return rewards

def experiment(args):
    # args = parser.parse_args()

    model_path = 'saved-models/expt-3-/G{}-P{}/{}'.format(args.sp_gamma, args.sp_percent, args.seed)
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_env = gym.make(args.randomized_env_id)

    if args.eval_env_id is None:
        args.eval_env_id = args.randomized_env_id

    rollout_env = make_vec_envs(args.eval_env_id, args.seed, N_PROCS)

    torch.manual_seed(args.seed)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Write Hyperparameters to file
    print("---------------------------------------")
    print("Current Arguments:")
    with open(os.path.join(model_path, "hps.txt"), 'w') as f:
        for arg in vars(args):
            print("{}: {}".format(arg, getattr(args, arg)))
            f.write("{}: {}\n".format(arg, getattr(args, arg)))
    print("---------------------------------------\n")

    alice_policy = AlicePolicy(state_dim=16)
    bob_policy = BobPolicy(state_dim=18, action_dim=2)
    alice_acting_policy = BobPolicy(state_dim=9, action_dim=2)

    # TODO: Keep changing this?
    alice_acting_policy.load_from_file('saved-models/expt-0b-reacher/good.pth')

    total_timesteps = 1e6
    timesteps = 0
    nepisodes = 0
    nselfplay = 0
    ntarget = 0
    best_reward = -np.inf

    learning_curve = []

    while timesteps < total_timesteps:
        if np.random.random() < args.sp_percent:
            # Alice
            training_env.seed(nselfplay)
            state = training_env.reset()

            alice_state = np.concatenate([state, np.zeros(STATE_DIM)])
            alice_done = False
            time_alice = 0
   
            while not alice_done and time_alice <= MAX_TIMESTEPS:
                action = old_bob_policy.select_action(alice_state, save_log_probs=False)
                state, reward, env_done, _ = training_env.step(action)
                alice_signal = alice_policy.select_action(alice_state)

                # TODO: Why does it end right away?
                alice_done = env_done or alice_signal

                if not alice_done: 
                    alice_state[:STATE_DIM] = state
                    alice_policy.log(0.)
                    time_alice += 1

            # Bob
            training_env.seed(nselfplay)
            state = training_env.reset()
            goal_state = alice_state[:STATE_DIM]

            bob_state = np.concatenate([state, goal_state])
            bob_done = False
            time_bob = 0

            while not bob_done and time_alice + time_bob <= MAX_TIMESTEPS:
                action = bob_policy.select_action(bob_state)

                state, reward, env_done, _ = training_env.step(action)
                bob_signal = check_closeness(state, goal_state)

                bob_done = env_done or bob_signal

                if not bob_done: 
                    bob_state[:STATE_DIM] = state
                    bob_policy.log(0.)
                    time_bob += 1

                else:
                    print(check_closeness(state, goal_state), time_bob, env_done, bob_signal)

            reward_alice = args.sp_gamma * max(0, time_bob - time_alice)
            reward_bob = -args.sp_gamma * time_bob

            print(time_alice, time_bob, 'rewards')

            alice_policy.log(reward_alice)
            bob_policy.log(reward_bob)

            nselfplay += 1
            alice_policy.finish_episode(gamma=0.99)
            bob_policy.finish_episode(gamma=0.99)

        else:
            state = training_env.reset()
            done = False
            bob_state = np.concatenate([state, np.zeros(STATE_DIM)])

            while not done:
                action = bob_policy.select_action(bob_state)

                state, reward, done, _ = training_env.step(action)
                bob_state[:STATE_DIM] = state
                bob_policy.log(reward)
                # SP is "free" (original paper)
                timesteps += 1

            ntarget += 1
            bob_policy.finish_episode(gamma=0.99)

        nepisodes += 1

        # Update old bob 
        # for param, target_param in zip(bob_policy.policy.parameters(), old_bob_policy.policy.parameters()):
        #     target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        if nepisodes % args.log_interval == 0:
            eval_rewards = evaluate_policy(rollout_env, bob_policy)
            eval_reward = np.mean(eval_rewards)
            learning_curve.append(eval_reward)

            torch.save(bob_policy.policy.state_dict(), 
                os.path.join('{}/{}.pth'.format(model_path, nepisodes)))

            if eval_reward > best_reward: 
                best_reward = eval_reward                

                torch.save(bob_policy.policy.state_dict(), 
                    os.path.join('{}/best.pth'.format(model_path)))

            print('Current: {}, Best: {}, Eps: {}, Timesteps: {}\n NSP: {}, NTARGET: {}'.format(
                eval_reward, best_reward, nepisodes, timesteps, nselfplay, ntarget)) 

    np.savez(os.path.join('{}/learningcurve.npz'.format(model_path)), learning_curve=learning_curve)

if __name__ == '__main__':
    seeds = [1, 2, 3]
    sp_gammas = [0.1, 0.25]
    sp_percents = [0.0, 0.01, 0.1, 0.25]

    for seed in seeds:
        for sp_gamma in sp_gammas:
            for sp_percent in sp_percents:
                args = Arguments(seed=seed, sp_gamma=sp_gamma, sp_percent=sp_percent)
                experiment(args)