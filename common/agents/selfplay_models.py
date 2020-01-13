import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli
import numpy as np

eps = np.finfo(np.float32).eps.item()


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