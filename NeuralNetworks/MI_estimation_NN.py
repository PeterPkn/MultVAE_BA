import torch
from torch import nn
from torch.nn import functional as F

class MI_net(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MI_net, self).__init__()
        self.mu_fc1 = nn.Linear(input_dim, hidden_dim)
        self.mu_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_fc3 = nn.Linear(hidden_dim, input_dim)

        self.logvar_fc1 = nn.Linear(input_dim, hidden_dim)
        self.logvar_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logvar_fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        x_mu = F.relu(self.mu_fc1(z))
        x_mu = F.relu(self.mu_fc2(x_mu))
        x_mu = self.mu_fc3(x_mu)
        x_logvar = F.relu(self.logvar_fc1(z))
        x_logvar = F.relu(self.logvar_fc2(x_logvar))
        x_logvar = self.logvar_fc3(x_logvar)
        return x_mu, x_logvar