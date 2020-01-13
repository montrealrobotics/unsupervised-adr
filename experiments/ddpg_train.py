import numpy as np
import torch
import gym
import argparse
import os
import gym_ergojr
from common.agents.ddpg.replay_buffer import ReplayBuffer
from common.agents.ddpg.ddpg import DDPG
from common.randomizer.wrappers import RandomizedEnvWrapper
from common.adr.adr import ADR
import multiprocessing as mp
from arguments import get_args


def get_env_params(env):
    observation = env.reset()
    # close the environment
    params = {'obs': observation["observation"].shape[0], 'goal': observation["achieved_goal"].shape[0], 'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0], 'max_timesteps': env._max_episode_steps}
    return params


args = get_args()
args.save_models = True
args.num_workers = 8

env = gym.make(args.env_name)
env_param = get_env_params(env)

# jobid = os.environ['SLURM_ARRAY_TASK_ID']
# seed = [40, 41, 42, 43, 44]
# args.seed = seed[int(jobid) - 1]  # Set seeds

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.nparticles = mp.cpu_count() - 1

state_dim = env_param["obs"]
action_dim = env_param["action"]
max_action = env_param["action_max"]
goal_dim = env_param["goal"]
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
env = RandomizedEnvWrapper(env, seed=args.seed)
env.reset()
svpg_rewards = []

# Initialize policy
policy = DDPG(args, state_dim, action_dim, max_action, goal_dim)

replay_buffer = ReplayBuffer(state_dim, action_dim)

# ADR integration
adr = ADR(
    nparticles=args.num_workers,
    nparams=args.n_params,
    state_dim=1,
    action_dim=1,
    temperature=10,
    svpg_rollout_length=args.svpg_rollout_length,
    svpg_horizon=25,
    max_step_length=0.05,
    reward_scale=1,
    initial_svpg_steps=0,
    seed=args.seed,
    discriminator_batchsz=320,
)
count = 0
args.save_dir = os.getcwd() + '/' + os.path.join(args.save_dir,
                              'sp{}polyak{}'.format(args.sp_percent, args.polyak) + '-' + args.approach,
                              str(args.seed))
model_path = os.path.join(args.save_dir, args.env_name)

if not os.path.isdir(model_path):
    os.makedirs(model_path)
args.save_dir = model_path
alice_envs = []
alice_envs_total = []

while total_timesteps < args.max_timesteps:
    if done:
        env.randomize([-1] * args.n_params)  # Randomize the environment
        if total_timesteps != 0:
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps, episode_num,
                                                                                episode_timesteps, episode_reward))
        # Reset environment
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    env_settings = adr.step_particles()
    env_settings = np.ascontiguousarray(env_settings)
    count = 0
    svpg_index = total_timesteps % args.svpg_rollout_length

    if np.random.random() < args.sp_percent:  # Self-play loop
        env.randomize(["default"] * args.n_params)

        bobs_goal_state, alice_time = policy.alice_loop(args, env, env_param)  # Alice Loop

        if total_timesteps % int(1e4) == 0:
            alice_envs_total.append(alice_envs)
            alice_envs = []

        if not args.only_sp:
            multiplier = np.clip(env_settings[svpg_index][0][:args.n_params], 0, 1.0)
            alice_envs.append(multiplier)
            env.randomize(multiplier)  # Randomize the env for bob
            bob_time, done = policy.bob_loop(env, env_param, bobs_goal_state, alice_time, replay_buffer)  # Bob Loop
            alice_reward = policy.train_alice(alice_time, bob_time)  # Train alice
            svpg_rewards.append(alice_reward)
            if len(svpg_rewards) == args.num_workers * args.svpg_rollout_length:  # ADR training
                all_rewards = np.reshape(np.asarray(svpg_rewards), (args.num_workers, args.svpg_rollout_length))
                adr._train_particles(all_rewards)
                svpg_rewards = []
        else:
            env.randomize(["default"] * args.n_params)
            bob_time, done = policy.bob_loop(env, env_param, bobs_goal_state, alice_time, replay_buffer)  # Bob Loop with only_sp=True
            alice_reward = policy.train_alice(alice_time, bob_time)
    else:
        observation = env.reset()
        obs = observation["observation"]
        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env_param['max_timesteps'] else float(done)
        done = done or episode_timesteps + 1 == env_param['max_timesteps']
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add(obs, action, new_obs["observation"], reward, done_bool)
        obs = new_obs["observation"]
        # Train the policy after collecting sufficient data
    if total_timesteps >= args.start_timesteps:
        policy.train(replay_buffer, args.batch_size)

    if timesteps_since_eval >= args.eval_freq:
        timesteps_since_eval %= args.eval_freq

        if args.save_models: policy.save(f'model_{total_timesteps}',
                                         directory=args.save_dir)
        np.save(f"{args.save_dir}/alice_envs.npy", alice_envs_total)
        print("---------------------------------------")
        print("Env Name: %s | Seed : %s | sp-percent : %s" % (args.env_name, args.seed, args.sp_percent))
        print("---------------------------------------")

    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

if args.save_models: policy.save(f'model_{total_timesteps}', directory=args.save_dir)
