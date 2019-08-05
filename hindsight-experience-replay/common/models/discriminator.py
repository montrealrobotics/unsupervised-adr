import torch
import torch.nn as nn

class MLPDiscriminator(nn.Module):
    """Discriminator class based on Feedforward Network
    Input is a state-action-state' transition
    Output is probability that it was from a reference trajectory
    """
    def __init__(self, state_dim, action_dim):
        super(MLPDiscriminator, self).__init__()
        
        self.l1 = nn.Linear((state_dim + action_dim + state_dim), 128)
        self.l2 = nn.Linear(128, 128)
        self.logic = nn.Linear(128, 1)
        
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    # Tuple of S-A-S'
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.logic(x)
        return torch.sigmoid(x)

class GAILMLPDiscriminator(nn.Module):
    """Discriminator class based on Feedforward Network
    Input is a state-action-state' transition
    Output is probability that it was from a reference trajectory
    """
    def __init__(self, state_dim, action_dim):
        super(GAILMLPDiscriminator, self).__init__()
        self.l1 = nn.Linear((state_dim + action_dim), 128)
        self.l2 = nn.Linear(128, 128)
        self.logic = nn.Linear(128, 1)
        
        self.logic.weight.data.mul_(0.1)
        self.logic.bias.data.mul_(0.0)

    # Tuple of S-A-S'
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = self.logic(x)
        return torch.sigmoid(x)