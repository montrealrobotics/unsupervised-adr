import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli

eps = np.finfo(np.float32).eps.item()

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(GaussianPolicy, self).__init__()

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class CategoricalPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CategoricalPolicy, self).__init__()
        self.net = nn.Linear(state_dim, action_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        action_scores = self.net(x)
        return F.softmax(action_scores, dim=1)

class BernoulliPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BernoulliPolicy, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.out = nn.Linear(32, 1)
        self.value = nn.Linear(32, 1)

        self.saved_log_probs = []
        self.rewards = []
        self.values = []

    def forward(self, x):
        x = self.base(x)
        termination_prob = self.out(x)
        value = self.value(x)
        return torch.sigmoid(termination_prob), value

class ADRPolicy:
    def __init__(self, state_dim=8, action_dim=10):
        self.policy = CategoricalPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)

    def select_action(self, state, deterministic=False, save_log_probs=True):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)

        if deterministic:
            action = m.mode()
        else:
            action = m.sample()
       
        if save_log_probs:
            self.policy.saved_log_probs.append(m.log_prob(action))

        return action.cpu().data.numpy()[0]

    def log(self, reward):
        self.policy.rewards.append(reward)

    def load_from_file(self, file):
        self.policy.load_state_dict(torch.load(file))

    def load_from_policy(self, original):
        with torch.no_grad():
            for param, target_param in zip(self.policy.parameters(), original.policy.parameters()):
                param.data.copy_(target_param.data)
                param.requires_grad = False

    def finish_episode(self, gamma):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)

        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

class BobPolicy:
    def __init__(self, state_dim=8, action_dim=2):
        self.policy = GaussianPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.args = args
    def select_action(self, state, deterministic=False, save_log_probs=True):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        # m = Categorical(probs)

        if deterministic:
            action = probs.mode()
        else:
            action = probs.sample()
       
        if save_log_probs:
            self.policy.saved_log_probs.append(probs.log_prob(action))

        return action.cpu().data.numpy()[0]

    def log(self, reward):
        self.policy.rewards.append(reward)

    def load_from_file(self, file):
        self.policy.load_state_dict(torch.load(file))

    def load_from_policy(self, original):
        with torch.no_grad():
            for param, target_param in zip(self.policy.parameters(), original.policy.parameters()):
                param.data.copy_(target_param.data)
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
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)

        rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]


class AlicePolicy:
    def __init__(self, state_dim=8, action_dim=1):
        self.policy = BernoulliPolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-3)
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
                param.data.copy_(target_param.data)
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
        self.optimizer.step()

        self.policy.rewards = []
        self.policy.saved_log_probs = []
        self.policy.values = []

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
        self.optimizer.step()

        self.policy.rewards = []
        self.policy.saved_log_probs = []
        self.policy.values = []
