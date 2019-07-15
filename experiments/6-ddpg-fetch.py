from policies.ddpg.ddpg import TD3, ReplayBuffer
import gym
import numpy as np
import torch

env = gym.make('FetchReach-v1')