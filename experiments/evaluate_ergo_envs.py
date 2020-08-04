import re
from common.randomizer.wrappers import RandomizedEnvWrapper
from locale import atof
import time
import torch
import common.randomizer as randomizer
from arguments import get_args
import gym
import gym_ergojr
import numpy as np
import os.path as osp
import os
import time
from common.agents.ddpg.ddpg import DDPG


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


if __name__ == '__main__':
    args = get_args()
    # load the model param
    # args.env_name = 'ErgoReacherRandomizedEnv-Headless-v0'
    # args.n_params = 1
    SEED = [i for i in range(111, 112)]
    APPROACH = ['unsupervised-adr']
    SP_PERCENT = [1.0]
    args.sp_polyak = 0.95
    for i, approach in enumerate(APPROACH):
        for seed in SEED:
            print('------------------------------------------------------------------------')
            print(f'Approach : {approach} | Seed : {seed} | Env : {args.env_name}')
            print('------------------------------------------------------------------------')
            args.save_dir = osp.join(os.getcwd()
                                     , '/saved_models', 'sp{}polyak{}'.
                                     format(SP_PERCENT[i], args.sp_polyak) + '-' + approach)

            model_path = osp.join(os.getcwd() + args.save_dir, str(seed), args.env_name)
            models_path = os.listdir(model_path + '/')

            # List the file names that start with model and sort according to number.
            models_path = list(filter(lambda x: x.lower().endswith(f'{args.sp_gamma}_actor.pth'), models_path))
            print(models_path)
            models_path.sort(key=natural_keys)
            print(models_path)
            env = gym.make("ErgoPushDefaultEnv-Headless-v0")
            env = RandomizedEnvWrapper(env, seed=12)
            env.randomize(['default'] * args.n_params)
            env.seed(args.seed + 10)
            eval_episodes = 1

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            max_action = float(env.action_space.high[0])
            goal_dim = 2
            final_dist = []
            policy = DDPG(args, state_dim, action_dim, max_action, goal_dim)
            for model in models_path:
                print(model)

                if not os.path.isfile(model_path + '/' + model):
                    continue
                policy.load(model, model_path)
                avg_reward = 0.
                avg_dist = 0
                model_dist = []
                for key in range(eval_episodes):
                    obs = env.reset()
                    done = False
                    while not done:
                        action = policy.select_action(np.array(obs["observation"], dtype=np.float64))
                        obs, reward, done, info = env.step(action)
                        avg_reward += reward

                    avg_dist += info["goal_dist"]
                final_dist.append(avg_dist/eval_episodes)
            np.save(model_path + f'{args.mode}_evaluation_{args.sp_gamma}.npy', np.asarray(final_dist))


