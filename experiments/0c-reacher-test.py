import gym
import pybulletgym
import time
import numpy as np


training_env = gym.make('ReacherPyBulletEnv-v0')
training_env.render(mode='human')
st = training_env.reset()
xy = training_env.robot.fingertip.pose().xyz()[:2]

for i in range(1000):
    training_env.step([1, -1])
    print('Time: {}, Distance: {}'.format(i, np.linalg.norm(training_env.robot.fingertip.pose().xyz()[:2] - xy)))
    training_env.render()
    time.sleep(0.05)


