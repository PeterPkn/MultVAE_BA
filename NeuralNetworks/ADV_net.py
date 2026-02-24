import torch
from torch import nn
from torch.nn import functional as F

class ADV_net(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ADV_net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.sigmoid(self.fc2(z))
        return z