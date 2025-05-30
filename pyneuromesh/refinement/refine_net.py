import torch
import torch.nn as nn

class RefineNet(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(self.model(x), min=0.001, max=1.0)

