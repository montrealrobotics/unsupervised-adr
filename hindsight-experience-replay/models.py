import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli

import os
from mpi4py import MPI

import numpy as np

eps = np.finfo(np.float32).eps.item()

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def setup_mpi(agent):
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'
  comm = MPI.COMM_WORLD
  param_vec = params_to_vec(agent, mode='params')
  comm.Bcast(param_vec, root=0)
  vec_to_params(param_vec, agent, mode='params')
  return comm

# Synchronises a network's gradients across processes
def sync_grads(comm, network):
  grad_vec_send = params_to_vec(network, mode='grads')
  grad_vec_recv = np.zeros_like(grad_vec_send)
  comm.Allreduce(grad_vec_send, grad_vec_recv, op=MPI.SUM)
  vec_to_params(grad_vec_recv / comm.Get_size(), network, mode='grads')

def params_to_vec(network, mode):
  attr = 'data' if mode == 'params' else 'grad'
  return np.concatenate([getattr(param, attr).detach().view(-1).numpy() for param in network.parameters()])


# Copies a numpy vector of parameters/gradients into a network
def vec_to_params(vec, network, mode):
  attr = 'data' if mode == 'params' else 'grad'
  param_pointer = 0
  for param in network.parameters():
    getattr(param, attr).copy_(torch.from_numpy(vec[param_pointer:param_pointer + param.data.numel()]).view_as(param.data))
    param_pointer += param.data.numel()


class BernoulliPolicyFetch(nn.Module):
    def __init__(self, goal_dim, action_dim=1):
        super(BernoulliPolicyFetch, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(goal_dim * 2, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
        )

        self.out = nn.Linear(300, 1)
        self.value = nn.Linear(300, 1)

        self.saved_log_probs = []
        self.rewards = []
        self.values = []

    def forward(self, x):
        x = self.base(x)
        termination_prob = self.out(x)
        value = self.value(x)
        return torch.sigmoid(termination_prob), value


class AlicePolicyFetch:
    def __init__(self, args, goal_dim, action_dim=1):
        self.policy = BernoulliPolicyFetch(goal_dim, action_dim=1)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-3)
        # self.comm = setup_mpi(self.policy)
        self.args = args

    def select_action(self, state, deterministic=False, save_log_probs=True):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, value = self.policy(state)
        m = Bernoulli(probs)

        action = m.sample()

        if save_log_probs:
            self.policy.saved_log_probs.append(m.log_prob(action))
            self.policy.values.append(value)

        return action.cpu().data.numpy()[0]

    def log(self, reward):
        self.policy.rewards.append(reward)

    def load_from_file(self, file):
        self.policy.load_state_dict(torch.load(file))

    def load_from_policy(self, original):
        with torch.no_grad():
            for param, target_param in zip(self.policy.parameters(), original.policy.parameters()):
                param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)
                param.requires_grad = False

    def perturb(self, alpha, weight_noise, bias_noise):
        with torch.no_grad():
            weight_noise = torch.from_numpy(alpha * weight_noise).float()
            bias_noise = torch.from_numpy(alpha * bias_noise).float()
            for param, noise in zip(self.policy.parameters(), [weight_noise, bias_noise]):
                param.add_(noise)
                param.requires_grad = False

    def finish_episode(self, gamma):
        R = 0
        policy_loss = []
        returns = []

        # Normalize
        self.policy.rewards = np.array(self.policy.rewards)
        self.policy.rewards = (self.policy.rewards - self.policy.rewards.mean()) / (self.policy.rewards.std() + eps)

        for r in self.policy.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, torch.FloatTensor([R]).unsqueeze(1))

        log_probs = torch.cat(self.policy.saved_log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.policy.values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        # sync_grads(self.comm, self.policy)
        self.optimizer.step()

        self.policy.rewards = []
        self.policy.saved_log_probs = []
        self.policy.values = []


# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.self_play = True
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.self_play = True
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value
