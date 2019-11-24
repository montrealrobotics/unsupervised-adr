import numpy as np
import gym_ergojr
from gym_ergojr.envs.ergo_pusher_env import ErgoPusherEnv

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class ErgoPusherRandomizedEnv(ErgoPusherEnv):
    def __init__(self, **kwargs):
        self.dimensions = []
        self.config_file = kwargs.get('config')
        del kwargs['config']

        super().__init__(**kwargs)
        self.reward_type = 'sparse'
        self.distance_threshold = 0.05

        # # this is affected by the DR
        # self.puck.friction

    def step(self, action):
        obs, reward, done, info = super().step(action)
        observation = {"observation": obs,
                       "achieved_goal": obs[6:8],
                       "desired_goal": obs[8:]}
        return observation, reward, done, info

    def update_randomized_params(self):

        self.puck.friction = self.dimensions[0].current_value

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d