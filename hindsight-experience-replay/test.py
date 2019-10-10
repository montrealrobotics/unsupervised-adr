import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper


env = gym.make("FetchSlideRandomizedEnv-v0")
env = RandomizedEnvWrapper(env, seed=12)
obs = env.reset()
env.randomize([0])

for i in range(1000):
    obs, _, _, _ = env.step(env.action_space.sample())
    env.render()
    if i % 100 == 0:
        env.reset()
