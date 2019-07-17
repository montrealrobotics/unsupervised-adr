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

env = gym.make(args.env_name)
obs = env.reset()

STATE_DIM = obs["observation"].shape[0]
GOAL_DIM = obs["achieved_goal"].shape[0]
TAU = 0.01

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
            action = alice_policy.select_action(alice_state)
            obs, reward, done, _ = env.step(action)
            
            #Stopping Criteria
            if action > np.random.random(): alice_done = True 
            alice_done = done or alice_time + 1 == max_timesteps
            if alice_done== False:
                alice_state[GOAL_DIM:] = obs["achieved_goal"]
                bobs_goal_state = obs["acieved_goal"]
                alice_time += 1
        # Bob's policy ...









