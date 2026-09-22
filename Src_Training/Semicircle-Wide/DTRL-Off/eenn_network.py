"""Three-port Actor + Twin-Q aligned with efull (Semicircle width: 256)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp(in_dim: int, hidden_sizes: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def squashed_gaussian_sample(
    mean: torch.Tensor, log_std_raw: torch.Tensor, log_std_min: float, log_std_max: float
) -> tuple[torch.Tensor, torch.Tensor]:
    log_std = torch.clamp(log_std_raw, log_std_min, log_std_max)
    std = log_std.exp()
    dist = Normal(mean, std)
    x_t = dist.rsample()
    action = torch.tanh(x_t)
    log_prob = dist.log_prob(x_t).sum(dim=-1, keepdim=True)
    log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    return action, log_prob


class SharedBackbone256(nn.Module):
    """obs → 256 + ReLU"""

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(inplace=True))
        self.out_dim = 256

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorEENN3Semicircle(nn.Module):
    """Semicircle joint SAC.

    - e1: obs → 256 → out (exit1 header is connected to backbone output)
    - e2: obs → 256 → 128 → 128 → out
    - efull (exit3): obs → 256 → 128 → 128 → 64 → 64 → out    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.backbone = SharedBackbone256(obs_dim)
        w = self.backbone.out_dim
        self.exit1_mean = nn.Linear(w, act_dim)
        self.exit1_log_std = nn.Linear(w, act_dim)
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.exit2_mean = nn.Linear(128, act_dim)
        self.exit2_log_std = nn.Linear(128, act_dim)
        self.trunk_exit3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.exit3_mean = nn.Linear(64, act_dim)
        self.exit3_log_std = nn.Linear(64, act_dim)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.backbone(obs)
        m1 = self.exit1_mean(h1)
        ls1 = self.exit1_log_std(h1)
        h2 = self.fc2(h1)
        h3 = self.fc3(h2)
        m2 = self.exit2_mean(h3)
        ls2 = self.exit2_log_std(h3)
        h4 = self.trunk_exit3(h3)
        m3 = self.exit3_mean(h4)
        ls3 = self.exit3_log_std(h4)
        return m1, ls1, m2, ls2, m3, ls3

    def sample_all(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m1, ls1, m2, ls2, m3, ls3 = self.forward(obs)
        a1, lp1 = squashed_gaussian_sample(m1, ls1, self.LOG_STD_MIN, self.LOG_STD_MAX)
        a2, lp2 = squashed_gaussian_sample(m2, ls2, self.LOG_STD_MIN, self.LOG_STD_MAX)
        a3, lp3 = squashed_gaussian_sample(m3, ls3, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return a1, lp1, a2, lp2, a3, lp3

    def joint_param_groups(
        self,
    ) -> tuple[
        list[nn.Parameter],
        list[nn.Parameter],
        list[nn.Parameter],
        list[nn.Parameter],
        list[nn.Parameter],
        list[nn.Parameter],
        list[nn.Parameter],
    ]:
        shared = list(self.backbone.parameters())
        fc2 = list(self.fc2.parameters())
        fc3 = list(self.fc3.parameters())
        deep = list(self.trunk_exit3.parameters())
        e1 = list(self.exit1_mean.parameters()) + list(self.exit1_log_std.parameters())
        e2 = list(self.exit2_mean.parameters()) + list(self.exit2_log_std.parameters())
        e3 = list(self.exit3_mean.parameters()) + list(self.exit3_log_std.parameters())
        return shared, fc2, fc3, deep, e1, e2, e3


class QNetworkFull(nn.Module):
    """Aligned with efull: (s,a) → 256 → 128 → 128 → 64 → 64 → 1"""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, (256, 128, 128, 64, 64), 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class CriticDeepTwin(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.q1 = QNetworkFull(obs_dim, act_dim)
        self.q2 = QNetworkFull(obs_dim, act_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)
