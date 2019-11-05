import randomizer
import gym
import numpy as np
from randomizer.wrappers import RandomizedEnvWrapper


env = gym.make("TwoNoisyFetchHookRandomEnv-v0")
env = RandomizedEnvWrapper(env, seed=12)
obs = env.reset()


for i in range(1000):
    obs, rew, _, _ = env.step(np.zeros((env.action_space.shape[0])))
    # print(rew)
    env.render()
    if i % 100 == 0:
        print(obs['observation'])
        print('done')
        env.randomize([1])
        env.reset()
