"""Three-port Actor + Twin-Q aligned with efull (Go2: 512 wide)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _mlp_elu(in_dim: int, hidden_sizes: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(d, h), nn.ELU(inplace=True)]
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


class SharedBackbone512(nn.Module):
    """obs → 512 + ELU"""

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 512), nn.ELU(inplace=True))
        self.out_dim = 512

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorEENN3Go2(nn.Module):
    """Three-port strategy (Go2 joint SAC).

    - e1 (exit1): obs → 512 → out
    - e2 (exit2): obs → 512 → 256 → 128 → out
    - efull (exit3): obs → 512 → 256 → 128 → 128 → 128 → out    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.backbone = SharedBackbone512(obs_dim)
        w = self.backbone.out_dim
        self.exit1_mean = nn.Linear(w, act_dim)
        self.exit1_log_std = nn.Linear(w, act_dim)
        self.fc256 = nn.Sequential(nn.Linear(w, 256), nn.ELU(inplace=True))
        self.fc128 = nn.Sequential(nn.Linear(256, 128), nn.ELU(inplace=True))
        self.exit2_mean = nn.Linear(128, act_dim)
        self.exit2_log_std = nn.Linear(128, act_dim)
        self.trunk_efull = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, 128),
            nn.ELU(inplace=True),
        )
        self.exit3_mean = nn.Linear(128, act_dim)
        self.exit3_log_std = nn.Linear(128, act_dim)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = self.backbone(obs)
        m1 = self.exit1_mean(h0)
        ls1 = self.exit1_log_std(h0)
        h1 = self.fc256(h0)
        h2 = self.fc128(h1)
        m2 = self.exit2_mean(h2)
        ls2 = self.exit2_log_std(h2)
        h3 = self.trunk_efull(h2)
        m3 = self.exit3_mean(h3)
        ls3 = self.exit3_log_std(h3)
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
        fc2 = list(self.fc256.parameters())
        fc3 = list(self.fc128.parameters())
        deep = list(self.trunk_efull.parameters())
        e1 = list(self.exit1_mean.parameters()) + list(self.exit1_log_std.parameters())
        e2 = list(self.exit2_mean.parameters()) + list(self.exit2_log_std.parameters())
        e3 = list(self.exit3_mean.parameters()) + list(self.exit3_log_std.parameters())
        return shared, fc2, fc3, deep, e1, e2, e3


class QNetworkEfull(nn.Module):
    """Aligned with efull: (s,a) → 512 → 256 → 128 → 128 → 128 → 1"""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = _mlp_elu(obs_dim + act_dim, (512, 256, 128, 128, 128), 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class CriticDeepTwin(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.q1 = QNetworkEfull(obs_dim, act_dim)
        self.q2 = QNetworkEfull(obs_dim, act_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)
