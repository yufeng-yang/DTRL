"""EENN strategy and value network: including two-port (ActorEENN/CriticEENN) and three-port (ActorEENN3/CriticEENN3) variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Reparametric sampling tanh-Gaussian, returns (action, log_prob)."""
    log_std = torch.clamp(log_std_raw, log_std_min, log_std_max)
    std = log_std.exp()
    dist = Normal(mean, std)
    x_t = dist.rsample()
    action = torch.tanh(x_t)
    log_prob = dist.log_prob(x_t).sum(dim=-1, keepdim=True)
    log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    return action, log_prob


class SharedBackbone(nn.Module):
    """obs → 128 (single hidden layer + ReLU) for the first segment of dual-port Actor/SAC."""

    def __init__(self, obs_dim: int, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, width), nn.ReLU(inplace=True))
        self.out_dim = width

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorEENN(nn.Module):
    """Gaussian strategy: shared backbone + early head + late head (each outputs mean, log_std)."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.backbone = SharedBackbone(obs_dim)
        z = self.backbone.out_dim
        self.early_mean = nn.Linear(z, act_dim)
        self.early_log_std = nn.Linear(z, act_dim)
        self.late_trunk = nn.Sequential(
            nn.Linear(z, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.late_mean = nn.Linear(64, act_dim)
        self.late_log_std = nn.Linear(64, act_dim)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the mean of early and late, log_std_raw."""
        z = self.backbone(obs)
        m_e = self.early_mean(z)
        ls_e = self.early_log_std(z)
        h = self.late_trunk(z)
        m_l = self.late_mean(h)
        ls_l = self.late_log_std(h)
        return m_e, ls_e, m_l, ls_l

    def sample_both(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Each head samples independently and returns a_e, log_pi_e, a_l, log_pi_l."""
        m_e, ls_e, m_l, ls_l = self.forward(obs)
        a_e, lp_e = squashed_gaussian_sample(
            m_e, ls_e, self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        a_l, lp_l = squashed_gaussian_sample(
            m_l, ls_l, self.LOG_STD_MIN, self.LOG_STD_MAX
        )
        return a_e, lp_e, a_l, lp_l

    def joint_param_groups(
        self,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
        """Shared backbone/early proprietary/late proprietary for split joint backhaul."""
        shared = list(self.backbone.parameters())
        early = list(self.early_mean.parameters()) + list(self.early_log_std.parameters())
        late = (
            list(self.late_trunk.parameters())
            + list(self.late_mean.parameters())
            + list(self.late_log_std.parameters())
        )
        return shared, early, late


class QNetworkEarly(nn.Module):
    """exit1 Q: (s,a) → 128 → 1 (aligned with policy exit1: obs-128-out)."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, (128,), 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class QNetworkDeepExit3(nn.Module):
    """Aligned with joint_train deep exit3: (s,a) → 128 → 128 → 128 → 64 → 64 → 1."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, (128, 128, 128, 64, 64), 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class QNetworkLate(nn.Module):
    """exit3 / full depth Q: (s,a) → 128 → 128 → 64 → 64 → 1 (aligned with policy exit3: obs-128-128-64-64-out)."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, (128, 128, 64, 64), 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class QNetworkExit2(nn.Module):
    """exit2 Q: (s,a) → 128 → 128 → 1 (aligned with policy exit2: obs-128-128-out)."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, (128, 128), 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class ActorSAC(nn.Module):
    """Standard SAC strategy (full depth consistent with exit3): obs→128→128→64→64→tanh Gaussian."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.backbone = SharedBackbone(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Linear(64, act_dim)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.backbone(obs)
        h2 = self.fc2(h1)
        h = self.trunk(h2)
        return self.mean(h), self.log_std(h)

    def sample(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action, log_prob (after tanh)."""
        m, ls = self.forward(obs)
        return squashed_gaussian_sample(
            m, ls, self.LOG_STD_MIN, self.LOG_STD_MAX
        )


class CriticSAC(nn.Module):
    """Twin Q, the structure is consistent with QNetworkLate (to facilitate comparison with the EEEN full branch)."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.q1 = QNetworkLate(obs_dim, act_dim)
        self.q2 = QNetworkLate(obs_dim, act_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)


class CriticDeepTwin(nn.Module):
    """Single Twin-Q (two-way MLP parameters are independent), the depth is consistent with exit3 of ActorEENN3Deep."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.q1 = QNetworkDeepExit3(obs_dim, act_dim)
        self.q2 = QNetworkDeepExit3(obs_dim, act_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)


class CriticEENN(nn.Module):
    """Four sets of Twin-Q that do not share each other: early uses QNetworkEarly×2, and late uses QNetworkLate×2."""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.q1_early = QNetworkEarly(obs_dim, act_dim)
        self.q2_early = QNetworkEarly(obs_dim, act_dim)
        self.q1_late = QNetworkLate(obs_dim, act_dim)
        self.q2_late = QNetworkLate(obs_dim, act_dim)

    def both(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        q1e = self.q1_early(obs, action)
        q2e = self.q2_early(obs, action)
        q1l = self.q1_late(obs, action)
        q2l = self.q2_late(obs, action)
        return (q1e, q1l), (q2e, q2l)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        return self.both(obs, action)


class ActorEENN3(nn.Module):
    """Three Gaussian strategies.

    - **backbone**: obs→128+ReLU → **h1**, all three subsequent sections depend on this section.
    - **exit1**: mean/log_std → **obs-128-out** on h1.
    - **fc2**: h1→128+ReLU → **h2**; **Only exit2 and exit3** pass through.
    - **exit2**: mean/log_std → **obs-128-128-out** on h2.
    - **trunk_exit3**: h2→64→64+ReLU (two segments); **only exit3** then mean/log_std → **obs-128-128-64-64-out**.

    ``joint_param_groups``: shared=backbone, mid=fc2, deep=trunk_exit3 (consistent with joint_train gradient assignment).    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.backbone = SharedBackbone(obs_dim)
        w = self.backbone.out_dim
        self.exit1_mean = nn.Linear(w, act_dim)
        self.exit1_log_std = nn.Linear(w, act_dim)
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
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
        m2 = self.exit2_mean(h2)
        ls2 = self.exit2_log_std(h2)
        h3 = self.trunk_exit3(h2)
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
        list[torch.nn.Parameter],
        list[torch.nn.Parameter],
        list[torch.nn.Parameter],
        list[torch.nn.Parameter],
        list[torch.nn.Parameter],
        list[torch.nn.Parameter],
    ]:
        """shared(backbone) / mid(fc2) / deep(trunk_exit3) / Gaussian header for each exit."""
        shared = list(self.backbone.parameters())
        mid = list(self.fc2.parameters())
        deep = list(self.trunk_exit3.parameters())
        e1 = list(self.exit1_mean.parameters()) + list(self.exit1_log_std.parameters())
        e2 = list(self.exit2_mean.parameters()) + list(self.exit2_log_std.parameters())
        e3 = list(self.exit3_mean.parameters()) + list(self.exit3_log_std.parameters())
        return shared, mid, deep, e1, e2, e3


class ActorEENN3Deep(nn.Module):
    """Three-exit strategy (consistent with joint_train): obs→128→128 followed by exit1;

    Then 128→128 gets h3, exit2; then 64→64 gets exit3. That is
    e1: obs-128-128-out; e2: obs-128-128-128-out; e3: obs-128-128-128-64-64-out.    """

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.act_dim = act_dim
        self.backbone = SharedBackbone(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.exit1_mean = nn.Linear(128, act_dim)
        self.exit1_log_std = nn.Linear(128, act_dim)
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
        h2 = self.fc2(h1)
        m1 = self.exit1_mean(h2)
        ls1 = self.exit1_log_std(h2)
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
        """shared(backbone) / fc2 / fc3 / trunk_exit3 / Each exit Gaussian header."""
        shared = list(self.backbone.parameters())
        fc2 = list(self.fc2.parameters())
        fc3 = list(self.fc3.parameters())
        deep = list(self.trunk_exit3.parameters())
        e1 = list(self.exit1_mean.parameters()) + list(self.exit1_log_std.parameters())
        e2 = list(self.exit2_mean.parameters()) + list(self.exit2_log_std.parameters())
        e3 = list(self.exit3_mean.parameters()) + list(self.exit3_log_std.parameters())
        return shared, fc2, fc3, deep, e1, e2, e3


class CriticEENN3(nn.Module):
    """Six-way Twin-Q: one pair each of exit1/2/3 **parameters are not shared with each other** (a total of 6 independent MLPs).

    If it is written as ``w1*L1 + w2*L2 + w3*L3`` during training, it is just the weighted sum of three **independent TD losses**,
    It is convenient to call ``loss.backward()`` once; it is not to merge multiple Qs into "a joint network".
    Each branch still predicts its own Q_k(s,a), which is aligned with the kth port of the strategy.    """

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.q1_exit1 = QNetworkEarly(obs_dim, act_dim)
        self.q2_exit1 = QNetworkEarly(obs_dim, act_dim)
        self.q1_exit2 = QNetworkExit2(obs_dim, act_dim)
        self.q2_exit2 = QNetworkExit2(obs_dim, act_dim)
        self.q1_exit3 = QNetworkLate(obs_dim, act_dim)
        self.q2_exit3 = QNetworkLate(obs_dim, act_dim)

    def triple(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        p1 = (self.q1_exit1(obs, action), self.q2_exit1(obs, action))
        p2 = (self.q1_exit2(obs, action), self.q2_exit2(obs, action))
        p3 = (self.q1_exit3(obs, action), self.q2_exit3(obs, action))
        return p1, p2, p3

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        return self.triple(obs, action)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)
