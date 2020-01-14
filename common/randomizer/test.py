import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper


env = gym.make("FetchPushRandomizedEnv-v0")
# env = RandomizedEnvWrapper(env, seed=12)
obs = env.reset()


for i in range(1000):
    obs, rew, _, _ = env.step(env.action_space.sample())
    env.render()
    if i % 100 == 0:
        # print('done')
        print(rew)
        # Randomize the environment
        env.randomize([-1])
        env.reset()
