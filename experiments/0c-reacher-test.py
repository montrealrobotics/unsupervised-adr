import gym
import pybulletgym
import time
import numpy as np
from envs import *
from envs.wrappers import RandomizedEnvWrapper

np.random.seed(1234)

env = gym.make('ReacherRandomized-v0')
env = RandomizedEnvWrapper(env=env, seed=0)

lengths = np.linspace(0, 1, 10)
for i in range(10):
    print(lengths[i])
    env.randomize(randomized_values=[lengths[i]])
    env.reset()
    for _ in range(200):
        obs, reward, done, _ = env.step(env.action_space.sample())
        env.render()
        


