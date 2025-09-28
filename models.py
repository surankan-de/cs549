# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallEmbed(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class MINE(nn.Module):
    """
    Donsker-Varadhan MINE estimator T_\theta(x,y) -> scalar
    We'll embed x and y separately and then combine.
    """
    def __init__(self, pt_dim, ct_dim, hidden=256):
        super().__init__()
        self.embed_x = SmallEmbed(pt_dim, out_dim=hidden//2)
        self.embed_y = SmallEmbed(ct_dim, out_dim=hidden//2)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x, y):
        ex = self.embed_x(x)
        ey = self.embed_y(y)
        combined = torch.cat([ex, ey], dim=-1)
        return self.net(combined).squeeze(-1)  # shape (batch,)

class Classifier(nn.Module):
    def __init__(self, pt_dim, ct_dim, hidden=256, num_classes=2):
        super().__init__()
        in_dim = pt_dim + ct_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, num_classes)
        )
    def forward(self, x, y):
        h = torch.cat([x, y], dim=-1)
        return self.net(h)
