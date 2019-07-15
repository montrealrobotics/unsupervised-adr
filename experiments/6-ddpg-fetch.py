from policies.ddpg.ddpg import DDPG, ReplayBuffer
import gym
import numpy as np
import torch

env = gym.make('FetchReach-v1')