import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper


env = gym.make("FetchSlideRandomizedEnv-v0")
env = RandomizedEnvWrapper(env, seed=12)
obs = env.reset()


for i in range(1000):
    obs, rew, _, _ = env.step(env.action_space.sample())
    print(rew)

    if i % 100 == 0:
        print('done')
        env.randomize([1])
        env.reset()
