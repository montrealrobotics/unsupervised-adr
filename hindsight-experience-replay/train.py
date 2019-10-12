import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from subprocess import CalledProcessError
from ddpg_agent import ddpg_agent
import random
import torch

from adr.adr import ADR
import gym_ergojr
from randomizer.wrappers import RandomizedEnvWrapper


"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


def get_rank():
    rank = MPI.COMM_WORLD.Get_rank()
    return rank


def launch(args):
    # create the ddpg_agent
    rank = get_rank()
    # print(f"Rank in her : {rank}")
    jobid = os.environ['SLURM_ARRAY_TASK_ID']
    env = gym.make(args.env_name)
    # get the environment parameters
    env_params = get_env_params(env)
    # print(jobid)
    seed = [20, 21, 22, 23, 24] * 2
    seed.sort()
    sp_percent = [0.0, 0.5] * 5
    args.sp_percent = sp_percent[int(jobid) - 1]
    # set random seeds for reproduce
    args.seed = seed[int(jobid) - 1]
    env.seed(args.seed + rank)
    approach = ['udr', 'adr'] * 5
    args.approach = approach[int(jobid) - 1]
    env = RandomizedEnvWrapper(env, seed=args.seed + rank)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    # if args.env_name.find('Push') != -1:
    #     env.set_friction(args.friction)

    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params)

    if rank == 0:
        adr = ADR(
            nparticles=MPI.COMM_WORLD.Get_size(),
            nparams=args.n_param,
            state_dim=1,
            action_dim=1,
            temperature=10,
            svpg_rollout_length=args.svpg_rollout_length,
            svpg_horizon=25,
            max_step_length=0.05,
            reward_scale=1,
            initial_svpg_steps=0,
            seed=rank + args.seed,
            discriminator_batchsz=320,
        )
    else:
        adr = None

    ddpg_trainer.learn(adr, args.svpg_rollout_length)

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
