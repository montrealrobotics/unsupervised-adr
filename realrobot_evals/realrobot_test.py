import h5py
import torch
import time
import numpy as np
import gym
import argparse
import os
import os.path as osp
from tqdm import tqdm, trange

from common.agents.ddpg.ddpg import DDPG
import gym_ergojr

parser = argparse.ArgumentParser(description='Real Robot Experiment Driver')

parser.add_argument('--nepisodes', type=int, default=25, help='Number of trials per *seed*')
parser.add_argument('--experiment-prefix', type=str, default='real', help='Prefix to append to logs')
parser.add_argument('--log-dir', type=str, default='/data/fgolemo/UADR-results/real-robot', help='Log Directory Prefix')
parser.add_argument('--model-dir', type=str, default='saved_models/', help='Model Directory Prefix')  # TODO
parser.add_argument('--cont', type=str, default='190329-180631', help='To continue existing file, enter timestamp here')
parser.add_argument('--cont', type=str, default='', help='To continue existing file, enter timestamp here')

args = parser.parse_args()

if len(args.cont) == 0:
    TIMESTAMP = time.strftime("%y%m%d-%H%M%S")
    file_flag = "w"

else:
    TIMESTAMP = args.cont
    file_flag = "r+"

file_path = "{}/{}-{}.hdf5".format(args.log_dir, args.experiment_prefix, TIMESTAMP)

MAX_EPISODE_STEPS = 100
EPISODES = args.nepisodes
SEED = ["31", "32", "33", "34", "35"]
# Policies to look for
policies = ['udr', 'adr-1.0', 'baseline-sp-1.0']

env_name = "ErgoReacher-MultiGoal-Live-v1"
npa = np.array

img_buffer = []

if not osp.exists(args.log_dir):
    os.makedirs(args.log_dir)


env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

n_episodes = 3  # num of episodes to run
max_timesteps = 100  # max timesteps in one episode
render = True  # render the environment
save_gif = False  # png images are saved in gif folder

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
goal_dim = 2

policy = DDPG(args, state_dim, action_dim, max_action, goal_dim)

obs = env.reset()


with h5py.File(file_path, file_flag) as f:
    for policy_type in tqdm(policies, desc="approaches"):
        for torque_idx, torque in enumerate(tqdm(SEED, desc="torques...")):

            model_path = osp.join(args.model_dir, policy_type, f"Variant-{torque}")
            print(model_path)
            no_models = len(os.listdir(model_path))

            if policy_type not in f:  # if dataset doesn't have these tables
                log_group = f.create_group(policy_type)
                rewards = log_group.create_dataset("rewards", (no_models, len(SEED), EPISODES, MAX_EPISODE_STEPS),
                                                   dtype=np.float32)
                distances = log_group.create_dataset("distances", (no_models, len(SEED), EPISODES, MAX_EPISODE_STEPS),
                                                     dtype=np.float32)
                trajectories = log_group.create_dataset("trajectories",
                                                        (no_models, len(SEED), EPISODES, MAX_EPISODE_STEPS, 24),
                                                        dtype=np.float32)
                imgs = log_group.create_dataset("images",
                                                (no_models, len(SEED), EPISODES, MAX_EPISODE_STEPS, 480, 640, 3),
                                                dtype=np.uint8, compression="lzf")
            else:  # if tables are in dataset, grab their pointers
                rewards = f.get(f"/{policy_type}/rewards")
                distances = f.get(f"/{policy_type}/distances")
                trajectories = f.get(f"/{policy_type}/trajectories")
                imgs = f.get(f"/{policy_type}/images")

            tqdm.write(f'Starting analysis of {policy_type} - variant {torque}')

            for model_idx, actorpth in enumerate(tqdm(os.listdir(model_path), desc="models....")):
                # Load model weights
                policy.load(os.path.join(model_path, actorpth))

                for ep_num in trange(EPISODES, desc="episodes.."):
                    non_zero_steps = np.count_nonzero(trajectories[model_idx, torque_idx, ep_num], axis=1)

                    if np.count_nonzero(non_zero_steps) == 0:
                        obs = env.reset()
                        done = False
                        cumulative = 0
                        counter = 0
                        img_buffer = []
                        while counter < MAX_EPISODE_STEPS:
                            with torch.no_grad():
                                # print(counter, obs)
                                action = policy.select_action(np.array(obs))

                            nobs, reward, _, misc = env.step(action)
                            cumulative += reward

                            trajectories[model_idx, torque_idx, ep_num, counter, :] = np.concatenate(
                                [obs, action, nobs])
                            rewards[model_idx, torque_idx, ep_num, counter] = reward
                            distances[model_idx, torque_idx, ep_num, counter] = misc["distance"]
                            img_buffer.append(np.copy(misc["img"]))

                            obs = np.copy(nobs)
                            counter += 1

                        imgs[model_idx, torque_idx, ep_num, :counter, :, :, :] = img_buffer

                    f.flush()

                env.reset()
