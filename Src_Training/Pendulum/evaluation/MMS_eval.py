"""MMS dynamic port selection evaluation: OWF e1 / MMS m2 e2 / SAC full ef.

All three ports are SAC, extract latent_pi + mu from zip, forward tanh+unscale (without predict).
The deadline is sampled every 10 steps, and the deepest outlet that meets the deadline is selected based on the p quantile of each port's latency buffer.

Run: python mms_evaluation.py"""

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
from stable_baselines3 import SAC

_DTRL = Path(__file__).resolve().parent.parent / "DTRL-Off"
sys.path.insert(0, str(_DTRL))

from action_utils import unscale_action  # noqa: E402

ENV_ID = "Pendulum-v1"
CPU_CORE = 2
WARMUP_STEPS = 200
N_SEEDS = 100
START_SEED = 42
DECISION_INTERVAL = 10
DEADLINE_LOW_MS = 0.05
DEADLINE_HIGH_MS = 0.072
BUFFER_SIZE = 100
P_ESTIMATE = 90
ANGLE_ERR_THRESH = 0.032 * np.pi
OMEGA_THRESH = 1.0

# (label, weight path, depth: the bigger, the deeper)
_EXITS: list[tuple[str, Path, int]] = [
    (
        "e1",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/SWI/runs/"
            "SWI_20260510_162815/weights/SWI_80000_steps.zip"
        ),
        1,
    ),
    (
        "e2",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/MMS/runs/"
            "MMS_20260512_050155/weights/MMS_80000_steps.zip"
        ),
        2,
    ),
    (
        "ef",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/Full/runs/"
            "Full_20260512_044600/weights/Full_80000_steps.zip"
        ),
        3,
    ),
]


class SacActor(nn.Module):
    """SAC actor：latent_pi → mu → tanh。"""

    def __init__(self, sac_actor: nn.Module) -> None:
        super().__init__()
        self.trunk = sac_actor.latent_pi
        self.head = sac_actor.mu

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.trunk(obs)))


@dataclass
class ExitActor:
    tag: str
    actor: SacActor
    weight_path: Path
    depth: int


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


def _build_exit(tag: str, path: Path, depth: int) -> ExitActor:
    if not path.is_file():
        raise FileNotFoundError(path)
    model = SAC.load(str(path), device="cpu")
    actor = SacActor(model.policy.actor).eval()
    del model
    return ExitActor(tag=tag, actor=actor, weight_path=path, depth=depth)


def _timed_infer(
    exit_actor: ExitActor, obs: np.ndarray, action_space: spaces.Box
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        scaled = exit_actor.actor(x).detach().cpu().numpy().reshape(-1)
        action = unscale_action(scaled, action_space)
    return action, (time.perf_counter() - t0) * 1e3


def _warmup_exit(
    exit_actor: ExitActor,
    buf: LatencyBuffer,
    obs_dim: int,
    action_space: spaces.Box,
) -> None:
    z = np.zeros(obs_dim, dtype=np.float32)
    for _ in range(WARMUP_STEPS):
        _, lat_ms = _timed_infer(exit_actor, z, action_space)
        buf.push(lat_ms)
    print(
        f"  [{exit_actor.tag}] warmup {WARMUP_STEPS} times | buffer={len(buf)} | "
        f"p{P_ESTIMATE}={buf.p_estimate_ms():.4f} ms"
    )


def _sample_deadline_ms(rng: np.random.Generator) -> float:
    return float(rng.uniform(DEADLINE_LOW_MS, DEADLINE_HIGH_MS))


def _obs_balanced(obs: np.ndarray) -> bool:
    theta = float(np.arctan2(obs[1], obs[0]))
    return abs(theta) <= ANGLE_ERR_THRESH and abs(float(obs[2])) <= OMEGA_THRESH


def _episode_success(step_balanced: list[bool]) -> bool:
    n = len(step_balanced)
    return n > 0 and all(step_balanced[n // 2 :])


def _select_exit(deadline_ms: float, buffers: dict[str, LatencyBuffer]) -> str:
    ordered = sorted(_EXITS, key=lambda x: x[2], reverse=True)
    for tag, _, _ in ordered:
        if buffers[tag].p_estimate_ms() <= deadline_ms:
            return tag
    return "e1"


def _run_episode(
    exits: dict[str, ExitActor],
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
    exit_counts = {tag: 0 for tag, _, _ in _EXITS}
    deadline_ms = DEADLINE_HIGH_MS
    active = "e1"
    last_action: np.ndarray | None = None
    done = False

    while not done:
        if step % DECISION_INTERVAL == 0:
            deadline_ms = _sample_deadline_ms(rng)
            active = _select_exit(deadline_ms, buffers)

        ex = exits[active]
        new_action, lat_ms = _timed_infer(ex, obs, env.action_space)
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
    for tag, path, _ in _EXITS:
        if not path.is_file():
            raise SystemExit(f"Not found [{tag}]: {path}")

    _setup_cpu()
    env = gym.make(ENV_ID)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_space = env.action_space
    assert isinstance(act_space, spaces.Box)
    env.close()

    exits: dict[str, ExitActor] = {}
    buffers: dict[str, LatencyBuffer] = {}
    for tag, path, depth in _EXITS:
        exits[tag] = _build_exit(tag, path, depth)
        buffers[tag] = LatencyBuffer()

    print("MMS dynamic port selection (e1 OWF / e2 MMS m2 / ef SAC full)")
    arch = {
        "e1": "obs→128→out (OWF)",
        "e2": "obs→128→128→128→out (MMS m2)",
        "ef": "obs→128→128→128→64→64→out (full)",
    }
    for tag, path, _ in _EXITS:
        print(f"  [{tag}] {path} | {arch[tag]}")
    print(
        f"\nForward: tanh+unscale (not predict)\n"
        f"deadline∈[{DEADLINE_LOW_MS},{DEADLINE_HIGH_MS}] ms | every{DECISION_INTERVAL} step reselect | "
        f"buffer={BUFFER_SIZE} p{P_ESTIMATE} | seeds {START_SEED}..{START_SEED + N_SEEDS - 1}\n"
        f"success: second half |θ|≤{np.degrees(ANGLE_ERR_THRESH):.2f}° and |θ̇|≤{OMEGA_THRESH} rad/s"
    )
    print(f"\nPreheat (each {WARMUP_STEPS} Second-rate):")
    for tag, _, _ in _EXITS:
        _warmup_exit(exits[tag], buffers[tag], obs_dim, act_space)

    seeds = list(range(START_SEED, START_SEED + N_SEEDS))
    results: list[EpisodeStats] = []
    print("\nFormal evaluation:")
    for seed in seeds:
        st = _run_episode(exits, buffers, ENV_ID, seed)
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
    total_exits = {tag: sum(r.exit_counts[tag] for r in results) for tag, _, _ in _EXITS}

    print(f"\n{'=' * 70}")
    print(f"comprehensive ({N_SEEDS} seeds) — MMS dynamic port selection:")
    print(f"  mean return     = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  mean hit rate   = {hit_rates.mean():.4f} ± {hit_rates.std(ddof=0):.4f}")
    print(f"  success rate    = {success_rate:.4f} ({sum(r.success for r in results)}/{N_SEEDS})")
    print(f"  total exit use  = {total_exits}")
    print(f"  buffer p{P_ESTIMATE} (ms) = ", end="")
    print(", ".join(f"{t}={buffers[t].p_estimate_ms():.4f}" for t, _, _ in _EXITS))


if __name__ == "__main__":
    main()
