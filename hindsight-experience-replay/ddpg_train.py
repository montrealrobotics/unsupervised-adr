import numpy as np
import torch
import gym
import argparse
import os
import gym_ergojr
from replay_buffer import ReplayBuffer

import OurDDPG
from randomizer.wrappers import RandomizedEnvWrapper
from adr.adr import ADR
import multiprocessing as mp


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    avg_default_dist = 0
    avg_hard_dist = 0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_default_dist += distance_evaluation(obs, default=True)
            avg_hard_dist += distance_evaluation(obs, default=False)
            avg_reward += reward

    avg_reward /= eval_episodes
    avg_default_dist /= eval_episodes
    avg_hard_dist /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward, avg_default_dist, avg_hard_dist

def distance_evaluation(obs, default=True):
    ag = obs[6:8]
    g = obs[8:]
    if default:
        env.randomize(["default"])
    else:
        env.randomize([0.9])
    return np.linalg.norm(ag - g)


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs.shape[0],
            'goal': obs[8:].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params


parser = argparse.ArgumentParser()
parser.add_argument("--policy_name", default="OurDDPG")  # Policy name
parser.add_argument("--env_name", default="ErgoPushRandomizedEnv-Headless-v0")  # OpenAI gym environment name
parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=1e4,
                    type=int)  # How many time steps purely random policy is run for
parser.add_argument("--eval_freq", default=5e3, type=float)  # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=float)  # Max time steps to run environment for
parser.add_argument("--save_models", default=True, action="store_true")  # Whether or not models are saved
parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=100, type=int)  # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
parser.add_argument("--nparticles", default=1, type=int)
parser.add_argument('--svpg-rollout-length', type=int, default=5)
parser.add_argument('--sp-percent', type=float, default=0.1, help='Self Play Percentage')
parser.add_argument('--n-params', type=int, default=1)
parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
parser.add_argument('--approach', type=str, default='baseline', help='Different approaches for experiments')
parser.add_argument('--sp-gamma', type=float, default=0.1, help='Self play gamma')
parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1, help='number of workers for trajectories sampling')


args = parser.parse_args()


file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
print("---------------------------------------")
print("Settings: %s" % (file_name))
print("---------------------------------------")
args.save_models = True
args.num_wrokers = 8


env = gym.make(args.env_name)
env_param = get_env_params(env)
#jobid = os.environ['SLURM_ARRAY_TASK_ID']
#args.seed += int(jobid)
# Set seeds
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
args.nparticles = mp.cpu_count() - 1

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
goal_dim = 2
env = RandomizedEnvWrapper(env, seed=args.seed)
svpg_rewards = []

# Initialize policy
policy = OurDDPG.DDPG(args, state_dim, action_dim, max_action, goal_dim)

replay_buffer = ReplayBuffer()

# Evaluate untrained policy
evaluations = [evaluate_policy(policy)]

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
# ADR integration
adr = ADR(
    nparticles=args.num_wrokers,
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
# if not os.path.isdir(args.save_dir):
#
#     os.makedirs(args.save_dir)
if not os.path.isdir(model_path):
    os.makedirs(model_path)
args.save_dir = model_path

while total_timesteps < args.max_timesteps:

    if done:

        if total_timesteps != 0:
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(total_timesteps, episode_num,
                                                                                episode_timesteps, episode_reward))


            if args.policy_name == "TD3":
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                             args.policy_noise, args.noise_clip, args.policy_freq)
            else:
                policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

        # Evaluate episode
        if timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(policy))

            if args.save_models: policy.save(file_name, directory=args.save_dir)
            np.save(f"{args.save_dir}/{file_name}", evaluations)

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

        multiplier = np.clip(env_settings[svpg_index][0][0], 0, 1.0)

        env.randomize([multiplier])  # Randomize the env for bob

        bob_time, done = policy.bob_loop(env, env_param,  bobs_goal_state, alice_time, replay_buffer)  # Bob Loop
        alice_reward = policy.train_alice(alice_time, bob_time)  # Train alice

        svpg_rewards.append(alice_reward)
        if len(svpg_rewards) == args.num_wrokers * args.svpg_rollout_length:  # ADR training
            all_rewards = np.reshape(np.asarray(svpg_rewards), (args.num_wrokers, args.svpg_rollout_length))
            adr._train_particles(all_rewards)
            svpg_rewards = []
    else:
        # Select action randomly or according to policy
        env.randomize([-1])
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
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1

# Final evaluation
evaluations.append(evaluate_policy(policy))
if args.save_models: policy.save("%s" % (file_name), directory="./saved_models")
np.save("./results/%s" % (file_name), evaluations)
