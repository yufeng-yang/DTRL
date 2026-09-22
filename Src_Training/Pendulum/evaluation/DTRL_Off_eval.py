"""EENN dynamic deadline + press p97 delay to choose the deepest exit: formal evaluation.

Every 10 steps: sampling deadline∈[0.05,0.072] ms, select the deepest outlet that meets the deadline based on p95 of each port’s latency buffer.
If the inference of a certain step times out (miss deadline), the step will repeat the previous step (if there is no history in the first step, the current inference result will still be used).
Independent buffer for each port (last 100 inferences, milliseconds). Bind the core, warmup each mouthful 200 times, and then run 10 seeds.

Run: python formal_evaluation.py"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
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
CPU_CORE = 2
WARMUP_STEPS = 200
N_SEEDS = 100
START_SEED = 42
DECISION_INTERVAL = 10
DEADLINE_LOW_MS = 0.05
DEADLINE_HIGH_MS = 0.075
BUFFER_SIZE = 100
P_ESTIMATE = 90
ANGLE_ERR_THRESH = 0.02 * np.pi  # Angular error threshold (0 for upright)
OMEGA_THRESH = 1.0  # |Angular velocity| Threshold (rad/s)


class Exit1(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.mean = nn.Linear(128, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.fc2(self.backbone(obs)))


class Exit2(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.mean = nn.Linear(128, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.fc3(self.fc2(self.backbone(obs))))


class Exit3(nn.Module):
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


# (label, module class, exit_id, depth: the bigger, the deeper)
_EXITS: list[tuple[str, type[nn.Module], int, int]] = [
    ("e1", Exit1, 1, 1),
    ("e2", Exit2, 2, 2),
    ("ef", Exit3, 3, 3),
]


class LatencyBuffer:
    """Record the latest buffer_size inference time (milliseconds), use p97 as an estimate."""

    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self._buf: deque[float] = deque(maxlen=capacity)

    def push(self, latency_ms: float) -> None:
        self._buf.append(latency_ms)

    def p97_estimate_ms(self) -> float:
        if not self._buf:
            return float("inf")
        return float(np.percentile(np.asarray(self._buf, dtype=np.float64), P_ESTIMATE))

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class EpisodeStats:
    seed: int
    return_: float
    n_steps: int
    deadline_hits: int
    success: bool
    exit_counts: dict[str, int] = field(default_factory=dict)


def _setup_cpu() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {CPU_CORE})
        print(f"[cpu] bind core {CPU_CORE}，affinity={sorted(os.sched_getaffinity(0))}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _build_exit(full: ActorEENN3Deep, exit_id: int, obs_dim: int, act_dim: int) -> nn.Module:
    row = next(x for x in _EXITS if x[2] == exit_id)
    net: nn.Module = row[1](obs_dim, act_dim)
    net.backbone.load_state_dict(full.backbone.state_dict())
    net.fc2.load_state_dict(full.fc2.state_dict())
    if exit_id == 1:
        net.mean.load_state_dict(full.exit1_mean.state_dict())
    elif exit_id == 2:
        net.fc3.load_state_dict(full.fc3.state_dict())
        net.mean.load_state_dict(full.exit2_mean.state_dict())
    else:
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
        scaled = _infer(actor, obs)
        action = unscale_action(scaled, action_space)
    return action, (time.perf_counter() - t0) * 1e3


def _warmup_exit(
    actor: nn.Module, buf: LatencyBuffer, obs_dim: int, action_space: spaces.Box, tag: str
) -> None:
    z = np.zeros(obs_dim, dtype=np.float32)
    for _ in range(WARMUP_STEPS):
        _, lat_ms = _timed_infer(actor, z, action_space)
        buf.push(lat_ms)
    print(f"  [{tag}] warmup {WARMUP_STEPS} times | buffer={len(buf)} | p97={buf.p97_estimate_ms():.4f} ms")


def _sample_deadline_ms(rng: np.random.Generator) -> float:
    return float(rng.uniform(DEADLINE_LOW_MS, DEADLINE_HIGH_MS))


def _obs_balanced(obs: np.ndarray) -> bool:
    """Pendulum obs = [cos θ, sin θ, θ̇]; upright target θ=0."""
    theta = float(np.arctan2(obs[1], obs[0]))
    angle_err = abs(theta)
    return angle_err <= ANGLE_ERR_THRESH and abs(float(obs[2])) <= OMEGA_THRESH


def _episode_success(step_balanced: list[bool]) -> bool:
    """Each step in the second half is successful if it meets the angular error and angular velocity thresholds."""
    n = len(step_balanced)
    if n == 0:
        return False
    start = n // 2
    second_half = step_balanced[start:]
    return len(second_half) > 0 and all(second_half)


def _select_exit(deadline_ms: float, buffers: dict[str, LatencyBuffer]) -> str:
    """Select the deepest exit that satisfies p97 estimate <= deadline; if not, use the shallowest e1."""
    ordered = sorted(_EXITS, key=lambda x: x[3], reverse=True)
    for tag, _, _, _ in ordered:
        if buffers[tag].p97_estimate_ms() <= deadline_ms:
            return tag
    return "e1"


def _run_episode(
    actors: dict[str, nn.Module],
    buffers: dict[str, LatencyBuffer],
    env_id: str,
    seed: int,
) -> EpisodeStats:
    env = gym.make(env_id)
    assert isinstance(env.action_space, spaces.Box)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    ep_ret = 0.0
    step = 0
    hits = 0
    step_balanced: list[bool] = []
    exit_counts = {tag: 0 for tag, _, _, _ in _EXITS}
    done = False
    deadline_ms = DEADLINE_HIGH_MS
    active = "e1"
    last_action: np.ndarray | None = None
    while not done:
        if step % DECISION_INTERVAL == 0:
            deadline_ms = _sample_deadline_ms(rng)
            active = _select_exit(deadline_ms, buffers)

        new_action, lat_ms = _timed_infer(actors[active], obs, env.action_space)
        buffers[active].push(lat_ms)
        exit_counts[active] += 1
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
        exit_counts=exit_counts,
    )


def main() -> None:
    if not JOINT_MODEL.is_file():
        raise SystemExit(f"Not found: {JOINT_MODEL}")

    _setup_cpu()
    ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
    env_id = ckpt.get("env_id", "Pendulum-v1")
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    env.close()

    full = ActorEENN3Deep(obs_dim, act_dim)
    full.load_state_dict(ckpt["actor"])
    full.eval()

    actors: dict[str, nn.Module] = {}
    buffers: dict[str, LatencyBuffer] = {}
    for tag, _, eid, _ in _EXITS:
        actors[tag] = _build_exit(full, eid, obs_dim, act_dim)
        buffers[tag] = LatencyBuffer()

    print(f"Joint weight: {JOINT_MODEL}")
    print(
        f"deadline∈[{DEADLINE_LOW_MS},{DEADLINE_HIGH_MS}] ms | every{DECISION_INTERVAL} step reselect | "
        f"buffer={BUFFER_SIZE} p{P_ESTIMATE} | seeds {START_SEED}..{START_SEED + N_SEEDS - 1}\n"
        f"success: each step in the second half |θ|≤{ANGLE_ERR_THRESH:.4f} rad ({np.degrees(ANGLE_ERR_THRESH):.2f}°) "
        f"And |θ̇|≤{OMEGA_THRESH} rad/s"
    )
    print(f"\nPreheat (each {WARMUP_STEPS} Second-rate):")
    probe = gym.make(env_id)
    act_space = probe.action_space
    assert isinstance(act_space, spaces.Box)
    probe.close()

    for tag, _, _, _ in _EXITS:
        _warmup_exit(actors[tag], buffers[tag], obs_dim, act_space, tag)

    seeds = list(range(START_SEED, START_SEED + N_SEEDS))
    results: list[EpisodeStats] = []
    print("\nFormal evaluation:")
    for seed in seeds:
        # Clear the buffer before each round and retain the statistics established during the warmup phase - the user requires warmup before the start.
        # The official game should continue to use the buffer after warmup without resetting it.
        st = _run_episode(actors, buffers, env_id, seed)
        hit_rate = st.deadline_hits / max(st.n_steps, 1)
        results.append(st)
        print(
            f"  seed={seed} return={st.return_:.4f} steps={st.n_steps} "
            f"hit_rate={hit_rate:.4f} success={st.success} exits={st.exit_counts}"
        )

    rets = np.array([r.return_ for r in results], dtype=np.float64)
    hit_rates = np.array(
        [r.deadline_hits / max(r.n_steps, 1) for r in results], dtype=np.float64
    )
    success_rate = float(np.mean([r.success for r in results]))
    total_exits = {tag: sum(r.exit_counts[tag] for r in results) for tag, _, _, _ in _EXITS}

    print(f"\n{'=' * 70}")
    print(f"comprehensive ({N_SEEDS} seeds):")
    print(f"  mean return     = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  mean hit rate   = {hit_rates.mean():.4f} ± {hit_rates.std(ddof=0):.4f}")
    print(f"  success rate    = {success_rate:.4f} ({sum(r.success for r in results)}/{N_SEEDS})")
    print(f"  total exit use  = {total_exits}")
    print(f"  buffer p97 (ms) = ", end="")
    print(", ".join(f"{t}={buffers[t].p97_estimate_ms():.4f}" for t, _, _, _ in _EXITS))


if __name__ == "__main__":
    main()
