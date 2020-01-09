import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from models import AlicePolicyFetch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)

class DDPG(object):
    def __init__(self, args, state_dim, action_dim, max_action, goal_dim, discount=0.99, tau=0.005):
        self.args = args
        self.goal_dim = goal_dim
        self.actor = Actor(state_dim, action_dim, max_action).to(device)  # Bob's policy
        self.actor_target = copy.deepcopy(self.actor)  # Alice acting policy
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)  # Bob's critic network
        self.critic_target = copy.deepcopy(self.critic)  # Alice's critic network
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.alice_policy = AlicePolicyFetch(self.args, goal_dim, action_dim=1) # Alice Policy

        self.discount = discount
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=100):

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def alice_loop(self, args, env, env_params):
        alice_done = False
        alice_time = 0
        env.randomize(["default"] * args.n_params)
        observation = env.reset()

        obs = observation["observation"]
        ag = observation["achieved_goal"]
        alice_state = np.concatenate([ag, np.zeros(ag.shape[0])])

        while not alice_done and (alice_time < env_params['max_timesteps']):
            observation = torch.FloatTensor(obs.reshape(1, -1)).to(device)
            pi = self.actor_target(observation)
            action = pi.cpu().data.numpy().flatten()

            observation_new, reward, env_done, _ = env.step(action)
            obs_new = observation_new
            ag_new = observation_new["achieved_goal"]
            alice_signal = self.alice_policy.select_action(alice_state)
            bobs_goal_state = ag_new

            # Stopping Criteria
            if alice_signal > np.random.random(): alice_done = True
            alice_done = env_done or alice_time + 1 == env_params['max_timesteps'] or alice_signal and alice_time >= 1
            if not alice_done:
                alice_state[env_params["goal"]:] = ag_new
                bobs_goal_state = ag_new
                alice_time += 1
                self.alice_policy.log(0.0)
                obs = obs_new["observation"]
                ag = ag_new

            return bobs_goal_state, alice_time

    def bob_loop(self, env, env_params, bobs_goal_state, alice_time, replay_buffer):
        observation = env.reset()
        obs = observation["observation"]
        bob_done = False
        bob_time = 0
        while not bob_done and alice_time + bob_time < env_params['max_timesteps']:
            action = self.select_action(obs)
            new_obs, reward, env_done, _ = env.step(action)
            bob_signal = self._check_closeness(new_obs["achieved_goal"], bobs_goal_state)
            bob_done = env_done or bob_signal or bob_done

            if not bob_done:
                replay_buffer.add(obs, action, new_obs["observation"], reward, bob_done)
                obs = new_obs["observation"]
                bob_time += 1

        return bob_time, bob_done

    def train_alice(self, alice_time, bob_time):
        reward_alice = self.args.sp_gamma * max(0, bob_time - alice_time)
        self.alice_policy.log(reward_alice)
        self.alice_policy.finish_episode(gamma=0.99)
        return reward_alice

    @staticmethod
    def _check_closeness(state, goal):
        dist = np.linalg.norm(state - goal)
        return dist < 0.025

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s' % (directory, filename), map_location=lambda storage, loc: storage))
        # self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location='cpu'))
