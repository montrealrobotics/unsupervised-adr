import re
from common.randomizer.wrappers import RandomizedEnvWrapper
import common.randomizer as randomizer
from arguments import get_args
import gym
import time
import numpy as np


args = get_args()
env = gym.make(args.env_name)
env = RandomizedEnvWrapper(env, seed=12)
env.randomize(['default'] * args.n_params)
env.reset()

timesteps = 1000

actions = [-1, 0.5, -1, 0.5]
for _ in range(timesteps):
    _, _, done, _ = env.step(env.action_space.sample())
    env.render()
    time.sleep(0.01)
    if done:
        env.reset()