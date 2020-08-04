import argparse

"""
Here are the param for the training

"""
def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--sp-polyak', type=float, default=0.1, help='Polyak Averaging Coefficient')
    parser.add_argument('--sp-gamma', type=float, default=0.1, help='Self play gamma')
    parser.add_argument('--sp-percent', type=float, default=1.0, help='Self Play Percentage')
    parser.add_argument('--friction', type=float, default=0.18, help='friction parameter to set')
    parser.add_argument('--approach', type=str, default='baseline', help='Different approaches for experiments')
    parser.add_argument('--svpg-rollout-length', type=int, default=5)
    parser.add_argument('--mode', type=str, default='default')
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument("--policy-name", default="OurDDPG")  # Policy name
    parser.add_argument("--env-name", default="ErgoPushRandomizedEnv-Headless-v0")  # OpenAI gym environment name
    parser.add_argument("--start-timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval-freq", default=5e3, type=float)  # How often (time steps) we evaluate
    parser.add_argument("--max-timesteps", default=1e6, type=float)  # Max time steps to run environment for
    parser.add_argument("--save-models", default=True, action="store_true")  # Whether or not models are saved
    parser.add_argument("--expl-noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy-noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise-clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy-freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--nparticles", default=1, type=int)
    parser.add_argument('--n-params', type=int, default=1)
    parser.add_argument('--only-sp', type=bool, action='store_true')
    parser.add_argument('--use-slurm', type=bool, action='store_true')


# Add more arguments if needed.

    args = parser.parse_args()

    return args
