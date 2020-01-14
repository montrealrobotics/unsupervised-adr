import random
import os
import gym
import numpy as np
from gym import spaces
from tqdm import tqdm
from gym_ergojr.sim.objects import Puck
import pybullet as p

PUSHER_GOAL_X = [-.2, -.1] # Get three points
PUSHER_GOAL_Y = [-.1, .05] # Get three points
# Define a circle equation with three points. After that choose four equidistant points on that circle.

circle1 = np.array([[-0.1, -0.1],
                    [-0.15, -0.1],
                    [-0.1, 0.05]])
shift = [-0.029, -0.0475, -0.066]


def define_circle(p1, p2, p3):
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)


def angle(line1, line2):
    return np.arccos(np.dot(line1, line2)/(np.linalg.norm(line1)*np.linalg.norm(line2)))


goal_pos = np.asarray([[-0.154, 0.0, 0], [-0.15, 0.032, 0], [-0.15, -0.032, 0], [-0.137, 0.062, 0],
[-0.137, -0.062, 0], [-0.117, 0.088, 0], [-0.117, -0.088, 0], [-0.172, 0.0, 0],
[-0.168, 0.032, 0], [-0.168, -0.032, 0],[-0.155, 0.062, 0],[-0.155, -0.062, 0], [-0.135, 0.088, 0],
[-0.135, -0.088, 0], [-0.191, 0.0, 0], [-0.191, -0.0, 0], [-0.186, 0.032, 0], [-0.186, -0.032, 0],
[-0.174, 0.062, 0], [-0.174, -0.062, 0], [-0.154, 0.088, 0], [-0.154, -0.08, 0]])  # Don't hard code. Change afterwards

puck_positions = np.array([[-0.07, 0.05], 
                           [-0.85, 0.65], 
                           [-0.10, 0.08]])
class PuckEval(Puck):
    def __init__(self, friction=0.4):
        super(PuckEval, self).__init__()
        self.goal = None

    def reset(self):
        super().reset()

    def hard_reset(self):
        super().hard_reset()

    def add_target(self):
        self.dbo.goal = goal_pos[int(os.environ['goal_index'])]
        self.obj_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.02, length=0.01, rgbaColor=[0, 1, 0, 1])
        self.target = p.createMultiBody(
            baseVisualShapeIndex=self.obj_visual, basePosition=self.dbo.goal)

