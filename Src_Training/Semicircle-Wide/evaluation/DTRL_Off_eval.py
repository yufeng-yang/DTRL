"""Semicircle v6 EENN dynamic deadline + select the deepest exit according to delay buffer (off-policy DTRL).

Environment: SafetyPointSemicircle0-v6
Every 10 steps: deadline∈[0.1, 0.16] ms uniform sampling, select ports according to the p quantile of each port buffer.
miss deadline → reuse the previous action. Each port first warms up for 1000 steps (without writing the buffer), then runs 1000 steps to fill the buffer, and then formally evaluates (single core).

Run: python dtrl_off_policy_eval.py"""

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

_BOARD = Path(__file__).resolve().parent.parent
_DTRL = _BOARD / "DTRL-Off"
sys.path.insert(0, str(_BOARD))
sys.path.insert(0, str(_DTRL))

import Semicircle_env  # noqa: F401
import safety_gymnasium  # noqa: E402

from action_utils import unscale_action  # noqa: E402
from eenn_network import ActorEENN3Semicircle, SharedBackbone256  # noqa: E402

JOINT_MODEL = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Wide/DTRL-Off/runs/"
    "DTRL-Off_20260517_205942/best_model/DTRL-Off_best_model.pt"
)
ENV_ID = "SafetyPointSemicircle0-v6"
CPU_CORE = 2
WARMUP_STEPS = 1000
N_SEEDS = 100
START_SEED = 42
DECISION_INTERVAL = 10
DEADLINE_LOW_MS = 0.08
DEADLINE_HIGH_MS = 0.16
BUFFER_SIZE = 1000
P_ESTIMATE = 97
SUCCESS_RETURN_THRESH = 9.0


class SafetyToGymnasiumWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["cost"] = float(cost)
        return obs, reward, terminated, truncated, info

    def get_wrapper_attr(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        inner_getter = getattr(self.env, "get_wrapper_attr", None)
        if callable(inner_getter):
            return inner_getter(name)
        return getattr(self.env, name)


class Exit1(nn.Module):
    """e1: obs → 256 → out"""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone256(obs_dim)
        self.mean = nn.Linear(self.backbone.out_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.backbone(obs))


class Exit2(nn.Module):
    """e2: obs → 256 → 128 → 128 → out"""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone256(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.mean = nn.Linear(128, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone(obs)
        return self.mean(self.fc3(self.fc2(h)))


class Exit3(nn.Module):
    """ef: obs → 256 → 128 → 128 → 64 → 64 → out"""

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone256(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(64, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone(obs)
        h = self.fc3(self.fc2(h))
        return self.mean(self.trunk(h))


_EXITS: list[tuple[str, type[nn.Module], int, int]] = [
    ("e1", Exit1, 1, 1),
    ("e2", Exit2, 2, 2),
    ("ef", Exit3, 3, 3),
]


class LatencyBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self._buf: deque[float] = deque(maxlen=capacity)

    def push(self, latency_ms: float) -> None:
        self._buf.append(latency_ms)

    def p_estimate_ms(self) -> float:
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
    goal_at_end: bool
    exit_counts: dict[str, int] = field(default_factory=dict)


def _setup_cpu(*, quiet: bool = False) -> None:
    """Bind single core + torch single thread (warmup / should be called before formal evaluation)."""
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {CPU_CORE})
        if not quiet:
            print(f"[cpu] bind core {CPU_CORE}，affinity={sorted(os.sched_getaffinity(0))}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _build_exit(full: ActorEENN3Semicircle, exit_id: int, obs_dim: int, act_dim: int) -> nn.Module:
    row = next(x for x in _EXITS if x[2] == exit_id)
    net: nn.Module = row[1](obs_dim, act_dim)
    net.backbone.load_state_dict(full.backbone.state_dict())
    if exit_id == 1:
        net.mean.load_state_dict(full.exit1_mean.state_dict())
    elif exit_id == 2:
        net.fc2.load_state_dict(full.fc2.state_dict())
        net.fc3.load_state_dict(full.fc3.state_dict())
        net.mean.load_state_dict(full.exit2_mean.state_dict())
    else:
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


def _run_env_steps(
    actor: nn.Module,
    buf: LatencyBuffer | None,
    env_id: str,
    action_space: spaces.Box,
    seed: int,
    n_steps: int,
    *,
    record_latency: bool,
) -> None:
    """Run n_steps steps in the real environment; write to buffer when record_latency=True."""
    _setup_cpu(quiet=True)
    env = _make_env(env_id)
    obs, _ = env.reset(seed=seed)
    done = False
    ep = 0
    n = 0
    while n < n_steps:
        if done:
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
            done = False
        action, lat_ms = _timed_infer(actor, obs, action_space)
        if record_latency and buf is not None:
            buf.push(lat_ms)
        n += 1
        obs, _, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
    env.close()


def _warmup_exit(
    actor: nn.Module,
    env_id: str,
    action_space: spaces.Box,
    tag: str,
    seed: int,
) -> None:
    """Real environment warmup, no buffer writing (only JIT/cache warmup)."""
    _run_env_steps(
        actor, None, env_id, action_space, seed, WARMUP_STEPS, record_latency=False
    )
    print(f"  [{tag}] warmup {WARMUP_STEPS} Step (real environment, do not write buffer, core={CPU_CORE})")


def _fill_buffer_exit(
    actor: nn.Module,
    buf: LatencyBuffer,
    env_id: str,
    action_space: spaces.Box,
    tag: str,
    seed: int,
) -> None:
    """In the real environment, run BUFFER_SIZE steps again to fill up the latency buffer."""
    _run_env_steps(
        actor, buf, env_id, action_space, seed, BUFFER_SIZE, record_latency=True
    )
    print(
        f"  [{tag}] fill buffer {BUFFER_SIZE} step | len={len(buf)} | "
        f"p{P_ESTIMATE}={buf.p_estimate_ms():.4f} ms"
    )


def _make_env(env_id: str) -> gym.Env:
    env = safety_gymnasium.make(env_id, render_mode=None)
    return SafetyToGymnasiumWrapper(env)


def _sample_deadline_ms(rng: np.random.Generator) -> float:
    return float(rng.uniform(DEADLINE_LOW_MS, DEADLINE_HIGH_MS))


def _goal_achieved(env: gym.Env) -> bool:
    """Safety-Gymnasium: goal is on Builder.task, not in Builder itself."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    task = getattr(base, "task", None)
    if task is not None and hasattr(task, "goal_achieved"):
        return bool(task.goal_achieved)
    if hasattr(base, "goal_achieved"):
        return bool(base.goal_achieved)
    return False


def _episode_success(goal_at_end: bool, ep_return: float) -> bool:
    """Success = reaching the end point and return>9 (points will be deducted if you hit hazard, it is difficult to satisfy both)."""
    return goal_at_end and ep_return > SUCCESS_RETURN_THRESH


def _format_buffer_estimates(buffers: dict[str, LatencyBuffer]) -> str:
    parts = [
        f"{tag}: p{P_ESTIMATE}={buffers[tag].p_estimate_ms():.4f}ms (n={len(buffers[tag])})"
        for tag, _, _, _ in _EXITS
    ]
    return " | ".join(parts)


def _select_exit(deadline_ms: float, buffers: dict[str, LatencyBuffer]) -> str:
    ordered = sorted(_EXITS, key=lambda x: x[3], reverse=True)
    for tag, _, _, _ in ordered:
        if buffers[tag].p_estimate_ms() <= deadline_ms:
            return tag
    return "e1"


def _run_episode(
    actors: dict[str, nn.Module],
    buffers: dict[str, LatencyBuffer],
    env_id: str,
    seed: int,
) -> EpisodeStats:
    _setup_cpu(quiet=True)
    env = _make_env(env_id)
    assert isinstance(env.action_space, spaces.Box)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    ep_ret = 0.0
    step = 0
    hits = 0
    exit_counts = {tag: 0 for tag, _, _, _ in _EXITS}
    deadline_ms = DEADLINE_HIGH_MS
    active = "e1"
    last_action: np.ndarray | None = None
    done = False

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
        done = bool(terminated or truncated)
        step += 1

    goal_at_end = _goal_achieved(env)
    env.close()
    return EpisodeStats(
        seed=seed,
        return_=ep_ret,
        n_steps=step,
        deadline_hits=hits,
        success=_episode_success(goal_at_end, ep_ret),
        goal_at_end=goal_at_end,
        exit_counts=exit_counts,
    )


def main() -> None:
    if not JOINT_MODEL.is_file():
        raise SystemExit(f"Not found: {JOINT_MODEL}")

    _setup_cpu()
    ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
    env_id = ckpt.get("env_id", ENV_ID)

    env = _make_env(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_space = env.action_space
    assert isinstance(act_space, spaces.Box)
    env.close()

    full = ActorEENN3Semicircle(obs_dim, act_dim)
    full.load_state_dict(ckpt["actor"])
    full.eval()

    actors: dict[str, nn.Module] = {}
    buffers: dict[str, LatencyBuffer] = {}
    for tag, _, eid, _ in _EXITS:
        actors[tag] = _build_exit(full, eid, obs_dim, act_dim)
        buffers[tag] = LatencyBuffer()

    print(f"Joint weight: {JOINT_MODEL}")
    print(f"environment: {env_id}")
    print(
        f"deadline∈[{DEADLINE_LOW_MS},{DEADLINE_HIGH_MS}] ms | every{DECISION_INTERVAL} step reselect | "
        f"buffer={BUFFER_SIZE} p{P_ESTIMATE} | seeds {START_SEED}..{START_SEED + N_SEEDS - 1}\n"
        f"success: reaches the end point and return>{SUCCESS_RETURN_THRESH}(Points will be deducted if you hit hazard)"
    )
    print(
        f"\nPreheat + fill buffer (real environment {env_id}, single core {CPU_CORE}):\n"
        f"  1) warmup {WARMUP_STEPS} step (do not write buffer)\n"
        f"  2) Run again {BUFFER_SIZE} Step by step fill buffer (capacity {BUFFER_SIZE}）"
    )
    for tag, _, _, _ in _EXITS:
        _warmup_exit(actors[tag], env_id, act_space, tag, START_SEED)
        _fill_buffer_exit(actors[tag], buffers[tag], env_id, act_space, tag, START_SEED)

    seeds = list(range(START_SEED, START_SEED + N_SEEDS))
    results: list[EpisodeStats] = []
    print("\nFormal evaluation:")
    for seed in seeds:
        st = _run_episode(actors, buffers, env_id, seed)
        hit_rate = st.deadline_hits / max(st.n_steps, 1)
        results.append(st)
        print(
            f"  seed={seed} return={st.return_:.4f} steps={st.n_steps} "
            f"goal={st.goal_at_end} hit_rate={hit_rate:.4f} success={st.success} "
            f"exits={st.exit_counts}"
        )
        print(f"    buffer estimate: {_format_buffer_estimates(buffers)}")

    rets = np.array([r.return_ for r in results], dtype=np.float64)
    steps = np.array([r.n_steps for r in results], dtype=np.float64)
    hit_rates = np.array(
        [r.deadline_hits / max(r.n_steps, 1) for r in results], dtype=np.float64
    )
    success_rate = float(np.mean([r.success for r in results]))
    total_exits = {tag: sum(r.exit_counts[tag] for r in results) for tag, _, _, _ in _EXITS}

    print(f"\n{'=' * 70}")
    print(f"comprehensive ({N_SEEDS} seeds) — Semicircle off-policy DTRL:")
    print(f"  mean return     = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  average step    = {steps.mean():.4f} ± {steps.std(ddof=0):.4f}")
    print(f"  mean hit rate   = {hit_rates.mean():.4f} ± {hit_rates.std(ddof=0):.4f}")
    print(f"  success rate    = {success_rate:.4f} ({sum(r.success for r in results)}/{N_SEEDS})")
    print(f"  total exit use  = {total_exits}")
    print(f"  buffer p{P_ESTIMATE} (ms) = ", end="")
    print(", ".join(f"{t}={buffers[t].p_estimate_ms():.4f}" for t, _, _, _ in _EXITS))


if __name__ == "__main__":
    main()
