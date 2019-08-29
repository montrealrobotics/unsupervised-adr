import torch
from models import actor
from arguments import get_args
import gym
import numpy as np
import os.path as osp
import os
from randomizer.wrappers import RandomizedEnvWrapper


# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # load the model param
    args.save_dir = osp.join('/saved_models', 'sp{}polyak{}'.format(args.sp_percent, args.sp_polyak) + '-' + args.approach)
    model_path = osp.join(os.getcwd() +  args.save_dir, str(args.seed), args.env_name, 'model.pt')
    print('Loading: {}'.format(model_path))
    o_mean, o_std, g_mean, g_std, model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # create the environment
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env = RandomizedEnvWrapper(env, seed=args.seed)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0], 
                  'goal': observation['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  'max_episode_steps': 50
                  }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()

    evals = []
    for friction in np.geomspace(0.18 * 0.01, 1, 10):
    # for friction in [0.18 * 0.01]:
        friction_evals = []
        env.randomize(["default", friction])
        env.seed(1000)
        print('\t\t##### Friction {} #####'.format(friction))
        for i in range(args.demo_length):
            observation = env.reset()
            # start to do the demo
            obs = observation['observation']
            g = observation['desired_goal']
            for t in range(env_params['max_episode_steps']):
                # env.render()
                # time.sleep(0.01)
                inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
                with torch.no_grad():
                    pi = actor_network(inputs)
                action = pi.detach().numpy().squeeze()
                # put actions into the environment
                # print(action)
                observation_new, reward, _, info = env.step(action)
                env.render()
                obs = observation_new['observation']
            print('the episode is: {}, is success: {}'.format(i, info['is_success']))
            friction_evals.append(info['is_success'])

        evals.append((friction, friction_evals))

    np.save(osp.join(os.getcwd() + args.save_dir, str(args.seed), '{}'.format(args.env_name), '{}_generalization.npy'.format(args.approach)), evals)

