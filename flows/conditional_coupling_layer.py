import torch
import torch.nn as nn

class ConditionalCouplingLayer(nn.Module):
    def __init__(self, dim, feature_dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.d = dim // 2

        self.scale_net = nn.Sequential(
            nn.Linear(self.d + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.d),
            nn.Tanh()
        )

        self.translate_net = nn.Sequential(
            nn.Linear(self.d + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.d)
        )

    def forward(self, x, features, reverse=False):
        x1, x2 = x[:, :self.d], x[:, self.d:]
        cond_input = torch.cat([x1, features], dim=1)

        if not reverse:
            s = self.scale_net(cond_input)
            t = self.translate_net(cond_input)

            z2 = x2 * torch.exp(s) + t
            z = torch.cat([x1, z2], dim=1)

            log_det = torch.sum(s, dim=1)
            return z, log_det
        else:
            s = self.scale_net(cond_input)
            t = self.translate_net(cond_input)

            x2 = (x2 - t) * torch.exp(-s)
            x = torch.cat([x1, x2], dim=1)

            log_det = -torch.sum(s, dim=1)
            return x, log_det