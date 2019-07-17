import argparse
import gym
import numpy as np
from itertools import count
import torch
from envs import *
from envs.wrappers import RandomizedEnvWrapper
from envs.randomized_vecenv import make_vec_envs
from policies.simple import BobPolicy, AlicePolicyFetch
from envs.heuristics.lunar.heuristics import heuristic, uncalibrated
import os
import psutil


parser = argparse.ArgumentParser(description="Experiments on Fetch")
parser.add_argument('--env-name', type=str, default='ResidualSlipperyPush-v0')
parser.add_argument('--seed', type=int, default=-1, metavar='N', help='random seed')
parser.add_argument('--polyak', type=int, default=0.05, help='Polyak Averaging Coefficient')
parser.add_argument('--sp-gamma', type=int, default=0.1, help='Self play gamma')

env = gym.make(args.env_name)
obs = env.reset()
STATE_DIM = obs["observation"].shape[0]
GOAL_DIM = obs["achieved_goal"].shape[0]

def soft_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - args.polyak) * target_param.data + args.polyak * param.data)

def check_closeness(state, goal):
    dist =  np.linalg.norm(state - goal)
    return dist < 0.025

def experiment(args):
    env.seed(args.seed)
    
    max_timesteps = 100
    total_episodes = 50000
    timesteps = 0
    nepisodes = 0
    nselfplay = 0
    ntarget = 0
    best_reward = -np.inf
    
    alice_policy = AlicePolicyFetch(goal_dim=GOAL_DIM)
    alice_action_policy = BobPolicy(goal_state=GOAL_DIM)
    bob_policy = BobPolicy(goal_state=GOAL_DIM)
    alice_acting_policy.load_from_policy(bob_policy)

    #Training Loop
    while ntarget < total_episodes:
        obs = env.reset()
        goal_state = obs["achieved_goal"]
        alice_state = np.concatenate([goal_state, np.zeros(GOAL_DIM)])
        alice_done = False
        alice_time = 0
        bobs_goal_state = None
        
        #Alice Stopping Policy
        while not alice_done and (alice_time < max_timesteps):
            action = alice_action_policy.select_action(alice_state)
            obs, reward, env_done, _ = env.step(action)
            alice_signal = alice_policy.select_action(alice_state)
            
            #Stopping Criteria
            if alice_signal > np.random.random(): alice_done = True
            alice_done = alice_done or env_done or alice_time + 1 == max_timesteps
            if alice_done == False:
                alice_state[GOAL_DIM:] = obs["achieved_goal"]
                bobs_goal_state = obs["achieved_goal"]
                alice_time += 1
                alice_policy.log(0.0)

        # Bob's policy
        obs = env.reset()
        state = obs["observation"]
        bob_state = np.concatenate([state, bobs_goal_state])
        bob_done = False
        bob_time = 0

        while not bob_done and alice_time + bob_time < max_timesteps:
            action = bob_policy.select_action(bob_state)
            obs, reward, env_done, _ = env.step(action)
            bob_signal = check_closeness(obs["achieved_goal"], bobs_goal_state)
         
            bob_done = bob_signal
            if bob_done == False:
                bob_state[:GOAL_DIM] = obs["achieved_goal"]
                bob_policy.log(0.0)
                bob_time += 1

        alice_policy.log(args.sp_gamma * max(0, bob_time - alice_time)) #Alice Reward
        bob_policy.log(-args.sp_gamma * bob_time) #Bob Reward

        alice_policy.finish_episode(gamma=0.99)
        bob_policy.finish_episode(gamma=0.99)
        
        #soft update
        # will include soft_update(alice_action_policy, bob_policy) after selfplay percent + if/else training loops
        ntarget += 1

















