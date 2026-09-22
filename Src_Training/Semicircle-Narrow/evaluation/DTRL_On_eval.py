"""Semicircle narrow (v5) On-policy DTRL dynamic port selection evaluation (same rules as dtrl_offpolicy_evaluation).

e1: PPO obs→256 | e2: PPO obs→256→128→128 | ef: PPO obs→256→128→128→64→64
Forward: Gaussian mean + clip (not predict). Warmup 1000 steps for each port + fill buffer 1000 steps + 100 seeds (single core).

Run: python dtrl_on_evaluation.py"""

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
from stable_baselines3 import PPO

_NARROW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_NARROW))

import Semicircle_env  # noqa: F401
import safety_gymnasium  # noqa: E402

ENV_ID = "SafetyPointSemicircle0-v5"
CPU_CORE = 2
WARMUP_STEPS = 1000
N_SEEDS = 100
START_SEED = 42
DECISION_INTERVAL = 10
DEADLINE_LOW_MS = 0.08
DEADLINE_HIGH_MS = 0.15
BUFFER_SIZE = 1000
P_ESTIMATE = 97

_EXITS: list[tuple[str, Path, int]] = [
    (
        "e1",
        Path(
"/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/DTRL-On/runs/DTRL-On_e1_20260516_050501/checkpoints/DTRL-On_e1_300000_steps.zip"
        ),
        1,
    ),
    (
        "e2",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/DTRL-On/runs/"
            "DTRL-On_e2_20260516_053958/best_model/DTRL-On_e2_best_model.zip"
        ),
        2,
    ),
    (
        "ef",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/DTRL-On/runs/"
            "DTRL-On_full_20260516_021458/best_model/DTRL-On_full_best_model.zip"
        ),
        3,
    ),
]


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


class PpoActor(nn.Module):
    def __init__(self, policy_net: nn.Sequential, action_net: nn.Linear) -> None:
        super().__init__()
        self.trunk = policy_net
        self.head = action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))


@dataclass
class ExitActor:
    tag: str
    actor: PpoActor
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
    goal_at_end: bool
    had_collision: bool
    collision_steps: int
    exit_counts: dict[str, int] = field(default_factory=dict)


def _setup_cpu(*, quiet: bool = False) -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {CPU_CORE})
        if not quiet:
            print(f"[cpu] bind core {CPU_CORE}，affinity={sorted(os.sched_getaffinity(0))}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _build_exit(tag: str, path: Path, depth: int) -> ExitActor:
    if not path.is_file():
        raise FileNotFoundError(path)
    model = PPO.load(str(path), device="cpu")
    actor = PpoActor(
        model.policy.mlp_extractor.policy_net,
        model.policy.action_net,
    ).eval()
    del model
    return ExitActor(tag=tag, actor=actor, weight_path=path, depth=depth)


def _timed_infer(
    exit_actor: ExitActor, obs: np.ndarray, action_space: spaces.Box
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean = exit_actor.actor(x).detach().cpu().numpy().reshape(-1)
        action = np.clip(mean, action_space.low, action_space.high).astype(np.float32)
    return action, (time.perf_counter() - t0) * 1e3


def _run_env_steps(
    exit_actor: ExitActor,
    buf: LatencyBuffer | None,
    env_id: str,
    action_space: spaces.Box,
    seed: int,
    n_steps: int,
    *,
    record_latency: bool,
) -> None:
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
        action, lat_ms = _timed_infer(exit_actor, obs, action_space)
        if record_latency and buf is not None:
            buf.push(lat_ms)
        n += 1
        obs, _, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
    env.close()


def _warmup_exit(
    exit_actor: ExitActor,
    env_id: str,
    action_space: spaces.Box,
    seed: int,
) -> None:
    _run_env_steps(
        exit_actor, None, env_id, action_space, seed, WARMUP_STEPS, record_latency=False
    )
    print(
        f"  [{exit_actor.tag}] warmup {WARMUP_STEPS} step "
        f"(Real environment, do not write buffer, core={CPU_CORE})"
    )


def _fill_buffer_exit(
    exit_actor: ExitActor,
    buf: LatencyBuffer,
    env_id: str,
    action_space: spaces.Box,
    seed: int,
) -> None:
    _run_env_steps(
        exit_actor, buf, env_id, action_space, seed, BUFFER_SIZE, record_latency=True
    )
    print(
        f"  [{exit_actor.tag}] fill buffer {BUFFER_SIZE} step | len={len(buf)} | "
        f"p{P_ESTIMATE}={buf.p_estimate_ms():.4f} ms"
    )


def _make_env(env_id: str) -> gym.Env:
    return SafetyToGymnasiumWrapper(safety_gymnasium.make(env_id, render_mode=None))


def _sample_deadline_ms(rng: np.random.Generator) -> float:
    return float(rng.uniform(DEADLINE_LOW_MS, DEADLINE_HIGH_MS))


def _goal_achieved(env: gym.Env) -> bool:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    task = getattr(base, "task", None)
    if task is not None and hasattr(task, "goal_achieved"):
        return bool(task.goal_achieved)
    if hasattr(base, "goal_achieved"):
        return bool(base.goal_achieved)
    return False


def _episode_success(goal_at_end: bool, had_collision: bool) -> bool:
    return goal_at_end and not had_collision


def _format_buffer_estimates(buffers: dict[str, LatencyBuffer]) -> str:
    return " | ".join(
        f"{tag}: p{P_ESTIMATE}={buffers[tag].p_estimate_ms():.4f}ms (n={len(buffers[tag])})"
        for tag, _, _ in _EXITS
    )


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
    _setup_cpu(quiet=True)
    env = _make_env(env_id)
    assert isinstance(env.action_space, spaces.Box)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    ep_ret = 0.0
    step = 0
    hits = 0
    collision_steps = 0
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

        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += float(reward)
        if float(info.get("cost", 0.0)) > 0:
            collision_steps += 1
        done = bool(terminated or truncated)
        step += 1

    goal_at_end = _goal_achieved(env)
    had_collision = collision_steps > 0
    env.close()
    return EpisodeStats(
        seed=seed,
        return_=ep_ret,
        n_steps=step,
        deadline_hits=hits,
        success=_episode_success(goal_at_end, had_collision),
        goal_at_end=goal_at_end,
        had_collision=had_collision,
        collision_steps=collision_steps,
        exit_counts=exit_counts,
    )


def main() -> None:
    for tag, path, _ in _EXITS:
        if not path.is_file():
            raise SystemExit(f"Not found [{tag}]: {path}")

    _setup_cpu()
    env = _make_env(ENV_ID)
    act_space = env.action_space
    assert isinstance(act_space, spaces.Box)
    env.close()

    arch = {
        "e1": "PPO obs→256→out",
        "e2": "PPO obs→256→128→128→out",
        "ef": "PPO obs→256→128→128→64→64→out",
    }

    exits: dict[str, ExitActor] = {}
    buffers: dict[str, LatencyBuffer] = {}
    print("Load three-port PPO:")
    for tag, path, depth in _EXITS:
        exits[tag] = _build_exit(tag, path, depth)
        buffers[tag] = LatencyBuffer()
        print(f"  [{tag}] {path} | {arch[tag]}")

    print(f"\nEnvironment: {ENV_ID}")
    print(
        f"deadline∈[{DEADLINE_LOW_MS},{DEADLINE_HIGH_MS}] ms | every{DECISION_INTERVAL} step reselect | "
        f"buffer={BUFFER_SIZE} p{P_ESTIMATE} | seeds {START_SEED}..{START_SEED + N_SEEDS - 1}\n"
        f"Forward: Gaussian mean + clip (not predict)\n"
        f"Success: Reach the end point without collision (any step info[cost]>0 is considered a collision)"
    )
    print(
        f"\nPreheat + fill buffer (real environment, single core {CPU_CORE}):\n"
        f"  1) warmup {WARMUP_STEPS} step (do not write buffer)\n"
        f"  2) Run again {BUFFER_SIZE} Fill buffer step by step"
    )
    for tag, _, _ in _EXITS:
        _warmup_exit(exits[tag], ENV_ID, act_space, START_SEED)
        _fill_buffer_exit(exits[tag], buffers[tag], ENV_ID, act_space, START_SEED)

    results: list[EpisodeStats] = []
    print("\nFormal evaluation:")
    for seed in range(START_SEED, START_SEED + N_SEEDS):
        st = _run_episode(exits, buffers, ENV_ID, seed)
        hit_rate = st.deadline_hits / max(st.n_steps, 1)
        results.append(st)
        print(
            f"  seed={seed} return={st.return_:.4f} steps={st.n_steps} "
            f"goal={st.goal_at_end} collision={st.had_collision} "
            f"hit_rate={hit_rate:.4f} success={st.success} exits={st.exit_counts}"
        )
        print(f"    buffer estimate: {_format_buffer_estimates(buffers)}")

    rets = np.array([r.return_ for r in results], dtype=np.float64)
    steps = np.array([r.n_steps for r in results], dtype=np.float64)
    hit_rates = np.array(
        [r.deadline_hits / max(r.n_steps, 1) for r in results], dtype=np.float64
    )
    success_rate = float(np.mean([r.success for r in results]))
    total_exits = {tag: sum(r.exit_counts[tag] for r in results) for tag, _, _ in _EXITS}

    print(f"\n{'=' * 70}")
    print(f"comprehensive ({N_SEEDS} seeds) — Semicircle narrow on-policy DTRL:")
    print(f"  mean return     = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  average step    = {steps.mean():.4f} ± {steps.std(ddof=0):.4f}")
    print(f"  mean hit rate   = {hit_rates.mean():.4f} ± {hit_rates.std(ddof=0):.4f}")
    print(f"  success rate    = {success_rate:.4f} ({sum(r.success for r in results)}/{N_SEEDS})")
    print(f"  total exit use  = {total_exits}")
    print(f"  buffer p{P_ESTIMATE} (ms) = ", end="")
    print(", ".join(f"{t}={buffers[t].p_estimate_ms():.4f}" for t, _, _ in _EXITS))


if __name__ == "__main__":
    main()
