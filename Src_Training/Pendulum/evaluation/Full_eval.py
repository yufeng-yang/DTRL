"""Always use EENN's ef egress (the deepest subnet, the same forward path as ef when dynamically selecting the egress).

There is no multi-exit switching; deadline/miss reuses the previous action/success rule and is consistent with dtrl_off_evaluation.
The weights come from the exit3 branch of the three-port joint best_model.pt (non-SB3 predict).

Run: python full_evaluation.py"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

_DTRL = Path(__file__).resolve().parent.parent / "DTRL-Off"
sys.path.insert(0, str(_DTRL))

from action_utils import unscale_action  # noqa: E402
from eenn_network import ActorEENN3Deep, SharedBackbone  # noqa: E402

JOINT_MODEL = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/DTRL-Off/runs/"
    "three_exits_joint/DTRL-Off_Pendulum-v1_20260512_054426/best/DTRL-Off_best_model.pt"
)
ENV_ID = "Pendulum-v1"
CPU_CORE = 2
WARMUP_STEPS = 200
N_SEEDS = 100
START_SEED = 42
DECISION_INTERVAL = 10
DEADLINE_LOW_MS = 0.05
DEADLINE_HIGH_MS = 0.075
ANGLE_ERR_THRESH = 0.02 * np.pi
OMEGA_THRESH = 1.0


class ExitEF(nn.Module):
    """obs → 128 → 128 → 128 → 64 → 64 → out（ef）"""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(64, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.trunk(self.fc3(self.fc2(self.backbone(obs)))))


@dataclass
class EpisodeStats:
    seed: int
    return_: float
    n_steps: int
    deadline_hits: int
    success: bool


def _setup_cpu() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {CPU_CORE})
        print(f"[cpu] bind core {CPU_CORE}，affinity={sorted(os.sched_getaffinity(0))}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _build_ef(full: ActorEENN3Deep, obs_dim: int, act_dim: int) -> nn.Module:
    net = ExitEF(obs_dim, act_dim)
    net.backbone.load_state_dict(full.backbone.state_dict())
    net.fc2.load_state_dict(full.fc2.state_dict())
    net.fc3.load_state_dict(full.fc3.state_dict())
    net.trunk.load_state_dict(full.trunk_exit3.state_dict())
    net.mean.load_state_dict(full.exit3_mean.state_dict())
    return net.eval()


def _infer(actor: nn.Module, obs: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    return torch.tanh(actor(x)).detach().numpy().flatten()


def _timed_infer(
    actor: nn.Module, obs: np.ndarray, action_space: spaces.Box
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        action = unscale_action(_infer(actor, obs), action_space)
    return action, (time.perf_counter() - t0) * 1e3


def _obs_balanced(obs: np.ndarray) -> bool:
    theta = float(np.arctan2(obs[1], obs[0]))
    return abs(theta) <= ANGLE_ERR_THRESH and abs(float(obs[2])) <= OMEGA_THRESH


def _episode_success(step_balanced: list[bool]) -> bool:
    n = len(step_balanced)
    return n > 0 and all(step_balanced[n // 2 :])


def _sample_deadline_ms(rng: np.random.Generator) -> float:
    return float(rng.uniform(DEADLINE_LOW_MS, DEADLINE_HIGH_MS))


def _warmup(actor: nn.Module, obs_dim: int, action_space: spaces.Box) -> None:
    z = np.zeros(obs_dim, dtype=np.float32)
    for _ in range(WARMUP_STEPS):
        _timed_infer(actor, z, action_space)
    print(f"  [ef] warmup {WARMUP_STEPS} times forward")


def _run_episode(actor: nn.Module, action_space: spaces.Box, seed: int) -> EpisodeStats:
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    ep_ret = 0.0
    step = 0
    hits = 0
    step_balanced: list[bool] = []
    deadline_ms = DEADLINE_HIGH_MS
    last_action: np.ndarray | None = None
    done = False

    while not done:
        if step % DECISION_INTERVAL == 0:
            deadline_ms = _sample_deadline_ms(rng)

        new_action, lat_ms = _timed_infer(actor, obs, action_space)
        if lat_ms <= deadline_ms:
            hits += 1
            action = new_action
            last_action = action
        elif last_action is not None:
            action = last_action
        else:
            action = new_action
            last_action = action

        obs, reward, terminated, truncated, _ = env.step(action)
        ep_ret += float(reward)
        step_balanced.append(_obs_balanced(obs))
        done = bool(terminated or truncated)
        step += 1

    env.close()
    return EpisodeStats(
        seed=seed,
        return_=ep_ret,
        n_steps=step,
        deadline_hits=hits,
        success=_episode_success(step_balanced),
    )


def main() -> None:
    if not JOINT_MODEL.is_file():
        raise SystemExit(f"Not found: {JOINT_MODEL}")

    _setup_cpu()
    ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
    env_id = ckpt.get("env_id", ENV_ID)
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_space = env.action_space
    assert isinstance(act_space, spaces.Box)
    env.close()

    full = ActorEENN3Deep(obs_dim, act_dim)
    full.load_state_dict(ckpt["actor"])
    actor = _build_ef(full, obs_dim, act_dim)

    print(f"Strategy: Fixed ef (EENN exit3)\nJoint weight: {JOINT_MODEL}")
    print(
        f"deadline∈[{DEADLINE_LOW_MS},{DEADLINE_HIGH_MS}] ms | every{DECISION_INTERVAL} step resampling | "
        f"miss→previous action | seeds {START_SEED}..{START_SEED + N_SEEDS - 1}"
    )
    print("\nPreheat:")
    _warmup(actor, obs_dim, act_space)

    seeds = list(range(START_SEED, START_SEED + N_SEEDS))
    results: list[EpisodeStats] = []
    print("\nFormal evaluation:")
    for seed in seeds:
        st = _run_episode(actor, act_space, seed)
        hit_rate = st.deadline_hits / max(st.n_steps, 1)
        results.append(st)
        print(
            f"  seed={seed} return={st.return_:.4f} steps={st.n_steps} "
            f"hit_rate={hit_rate:.4f} success={st.success}"
        )

    rets = np.array([r.return_ for r in results], dtype=np.float64)
    hit_rates = np.array(
        [r.deadline_hits / max(r.n_steps, 1) for r in results], dtype=np.float64
    )
    success_rate = float(np.mean([r.success for r in results]))

    print(f"\n{'=' * 70}")
    print(f"comprehensive ({N_SEEDS} seeds) — fixed ef:")
    print(f"  mean return     = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  mean hit rate   = {hit_rates.mean():.4f} ± {hit_rates.std(ddof=0):.4f}")
    print(f"  success rate    = {success_rate:.4f} ({sum(r.success for r in results)}/{N_SEEDS})")


if __name__ == "__main__":
    main()
