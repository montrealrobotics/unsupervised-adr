import torch
import os
import os.path as osp
from datetime import datetime
import numpy as np
from mpi4py import MPI
from models import actor, critic
from utils import sync_networks, sync_grads
from replay_buffer import replay_buffer, ReplayBufferSelfPlay
from normalizer import normalizer
from her import her_sampler
from models import AlicePolicyFetch

"""
ddpg with HER (MPI-version)
"""


class ddpg_agent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        self.alice_policy = AlicePolicyFetch(self.args, goal_dim=env_params["goal"], action_dim=1)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        # her sampler
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        self.replay_buffer = ReplayBufferSelfPlay(capacity=int(1e6))
        # create the normalizer
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        # create the dict for store the model

        self.args.save_dir = osp.join(self.args.save_dir, 'sp{}polyak{}'.format(args.sp_percent, args.polyak), str(args.seed))

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.makedirs(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            # self.model_path = osp.join(self.model_path, str(args.seed))
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)

    def learn(self, adr, svpg_rollout_length):
        """
        train the network
        """
        evals = []
        rank = MPI.COMM_WORLD.Get_rank()
        comm = MPI.COMM_WORLD

        # start to collect samples
        for epoch in range(self.args.n_epochs):
            print('Epoch', epoch)
            alice_goals = []
            random_sp_arr = np.empty(self.args.n_cycles)

            if rank == 0:
                random_sp_arr = np.random.random(self.args.n_cycles)
                            
            comm.Bcast(random_sp_arr, root=0)

            for cycle in range(self.args.n_cycles):
                # set, broadcast the environments here (rollout particles)
                mb_obs, mb_ag, mb_g, mb_actions, mb_done = [], [], [], [], []
                is_sp_cycle = random_sp_arr[cycle] < self.args.sp_percent                       

                for i in range(self.args.num_rollouts_per_mpi):
                    svpg_index = i % svpg_rollout_length
                    if is_sp_cycle and svpg_index == 0:
                        if rank == 0:
                            env_settings = adr.step_particles()[:, :, 0]
                        else:
                            env_settings = np.empty((svpg_rollout_length, self.args.nmpi))

                        comm.Bcast(env_settings, root=0)
                        svpg_rewards = []

                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions, ep_done = [], [], [], [], []
                    # reset the environment
                    self.env.seed(rank + epoch * cycle + i + self.args.seed)
                    # TODO: Fix with sharath
                    
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples

                    if is_sp_cycle:
                        self.env.set_friction(1.0)
                        
                        alice_done = False
                        alice_time = 0
                        alice_state = np.concatenate([ag, np.zeros(self.env_params["goal"])])
                        # Alice Stopping Policy
                        while not alice_done and (alice_time < self.env_params['max_timesteps']):
                            with torch.no_grad():
                                input_tensor = self._preproc_inputs(obs, g)
                                pi = self.actor_target_network(input_tensor)
                                action = self._select_actions(pi)
                            observation_new, reward, env_done, _ = self.env.step(action)
                            obs_new = observation_new['observation']
                            ag_new = observation_new['achieved_goal']
                            alice_signal = self.alice_policy.select_action(alice_state)

                            # Stopping Criteria
                            if alice_signal > np.random.random(): alice_done = True
                            alice_done = env_done or alice_time + 1 == self.env_params[
                                'max_timesteps'] or alice_signal and alice_time >= 1
                            if not alice_done:
                                alice_state[self.env_params["goal"]:] = ag_new
                                bobs_goal_state = ag_new
                                alice_time += 1
                                self.alice_policy.log(0.0)
                                obs = obs_new
                                ag = ag_new

                        # Bob's policy
                        friction_multiplier = np.clip(env_settings[svpg_index][rank], 0.1, 0.9)
                        self.env.set_friction(friction_multiplier)
                        self.env.seed(rank + epoch * cycle + i + self.args.seed)
                        observation = self.env.reset()
                        obs = observation['observation']
                        ag = observation['achieved_goal']
                        
                        if rank == 0:
                            alice_goals.append(bobs_goal_state)
                        
                        bob_state = np.concatenate([obs, bobs_goal_state])
                        bob_done = False
                        bob_time = 0
                        for t in range(self.env_params['max_timesteps']):
                            with torch.no_grad():
                                input_tensor = self._preproc_inputs(obs, bobs_goal_state)
                                pi = self.actor_network(input_tensor)
                                action = self._select_actions(pi)

                            observation_new, reward, env_done, _ = self.env.step(action)

                            bob_signal = self._check_closeness(observation_new["achieved_goal"], bobs_goal_state)
                            bob_done = env_done or bob_signal or bob_done

                            if not bob_done:
                                bob_state[:self.env_params["goal"]] = ag_new
                                bob_time += 1
                                
                            obs_new = observation_new['observation']
                            ag_new = observation_new['achieved_goal']

                            # re-assign the observation\
                            ep_obs.append(obs.copy())
                            ep_ag.append(ag.copy())
                            ep_g.append(g.copy())
                            ep_actions.append(action.copy())
                            
                            obs = obs_new
                            ag = ag_new

                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        mb_obs.append(ep_obs)
                        mb_ag.append(ep_ag)
                        mb_g.append(ep_g)
                        mb_actions.append(ep_actions)
                        
                        reward_alice = self.args.sp_gamma * max(0, bob_time - alice_time)
                        svpg_rewards.append(reward_alice)
                        self.alice_policy.log(reward_alice)
                        self.alice_policy.finish_episode(gamma=0.99)

                        if (i + 1) % svpg_rollout_length == 0:
                            all_rewards = None
                            if rank == 0:
                                all_rewards = np.zeros((self.args.nmpi, svpg_rollout_length))

                            comm.Gather(np.array(svpg_rewards), all_rewards, root=0)
                            
                            if rank == 0:
                                adr._train_particles(all_rewards)

                            if rank == 0:
                                wait_hack = np.random.random(2)
                            else:
                                wait_hack = np.empty(2)

                            # Trick to sync
                            comm.Bcast(wait_hack, root=0)
                            

                    else:
                        for t in range(self.env_params['max_timesteps']):
                            with torch.no_grad():
                                input_tensor = self._preproc_inputs(obs, g)
                                pi = self.actor_network(input_tensor)
                                action = self._select_actions(pi)
                            # feed the actions into the environment
                            observation_new, _, done, info = self.env.step(action)
                            obs_new = observation_new['observation']
                            ag_new = observation_new['achieved_goal']
                            # append rollouts
                            ep_obs.append(obs.copy())
                            ep_ag.append(ag.copy())
                            ep_g.append(g.copy())
                            ep_actions.append(action.copy())
                            # re-assign the observation
                            obs = obs_new
                            ag = ag_new
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        mb_obs.append(ep_obs)
                        mb_ag.append(ep_ag)
                        mb_g.append(ep_g)
                        mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()

                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)

            # start to do the evaluation
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                evals.append(success_rate)
                np.save(osp.join(self.model_path, 'evals.npy'), evals)
                np.save(osp.join(self.model_path, 'alice_goals_Ep{}.npy'.format(epoch)), alice_goals)
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                            self.actor_network.state_dict()], self.model_path + '/model.pt')
    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                           size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # The shape of obs : (50, 10)
        # The shape of g: (50, 3)
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update

        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    def _check_closeness(self, state, goal):
        dist = np.linalg.norm(state - goal)
        return dist < 0.025

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # The Shape of obs : (256, 10)
        # The Shape of g: (256, 3)
        # The Shape of ag : (256, 3)
        # The Shape of actions: (256, 4)

        # pre-process the observation and goal
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()