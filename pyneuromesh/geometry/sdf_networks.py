import torch
import torch.nn as nn

class SDFNetwork(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128, num_layers=4):
        super(SDFNetwork, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, 1))  # Output: signed distance
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

