"""Semicircle narrow (v5) Fixed full SAC formal evaluation (same timing constraints as dtrl_offpolicy_evaluation).

Strategy: b1_full_only deep SAC (obs→256→128→128→64→64→out)
Every 10 steps: deadline∈[0.08, 0.15] ms; miss → reuse the previous step action.
Forward: tanh+unscale, no need to predict. Real environment warmup for 1000 steps.

Success: Reach the end point without collision (any step info[cost]>0 is considered a collision)

Run: python full_eval.py"""

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
from stable_baselines3 import SAC

_NARROW = Path(__file__).resolve().parent.parent
_DTRL = _NARROW / "DTRL-Off"
sys.path.insert(0, str(_NARROW))
sys.path.insert(0, str(_DTRL))

import Semicircle_env  # noqa: F401
import safety_gymnasium  # noqa: E402

from action_utils import unscale_action  # noqa: E402

MODEL_ZIP = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/Full/runs/"
    "Full_20260512_115359/best_model/Full_best_model.zip"
)
ENV_ID = "SafetyPointSemicircle0-v5"
CPU_CORE = 2
WARMUP_STEPS = 1000
N_SEEDS = 100
START_SEED = 42
DECISION_INTERVAL = 10
DEADLINE_LOW_MS = 0.08
DEADLINE_HIGH_MS = 0.15


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


class SacFullActor(nn.Module):
    """SAC full：latent_pi → mu → tanh。"""

    def __init__(self, sac_actor: nn.Module) -> None:
        super().__init__()
        self.trunk = sac_actor.latent_pi
        self.head = sac_actor.mu

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.trunk(obs)))


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


def _build_actor(model_zip: Path) -> tuple[SacFullActor, str]:
    if not model_zip.is_file():
        raise FileNotFoundError(model_zip)
    model = SAC.load(str(model_zip), device="cpu")
    actor = SacFullActor(model.policy.actor).eval()
    env_id = (
        model.env.spec.id
        if model.env is not None and hasattr(model.env, "spec") and model.env.spec
        else ENV_ID
    )
    del model
    return actor, env_id


def _timed_infer(
    actor: SacFullActor, obs: np.ndarray, action_space: spaces.Box
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        scaled = actor(x).detach().cpu().numpy().reshape(-1)
        action = unscale_action(scaled, action_space)
    return action, (time.perf_counter() - t0) * 1e3


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


def _warmup(actor: SacFullActor, env_id: str, action_space: spaces.Box, seed: int) -> None:
    _setup_cpu(quiet=True)
    env = _make_env(env_id)
    obs, _ = env.reset(seed=seed)
    done = False
    ep = 0
    n = 0
    while n < WARMUP_STEPS:
        if done:
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
            done = False
        action, _ = _timed_infer(actor, obs, action_space)
        n += 1
        obs, _, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
    env.close()
    print(f"  warmup {WARMUP_STEPS} step (real environment, core={CPU_CORE})")


def _run_episode(
    actor: SacFullActor, env_id: str, action_space: spaces.Box, seed: int
) -> EpisodeStats:
    _setup_cpu(quiet=True)
    env = _make_env(env_id)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    ep_ret = 0.0
    step = 0
    hits = 0
    collision_steps = 0
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
    )


def main() -> None:
    if not MODEL_ZIP.is_file():
        raise SystemExit(f"Not found: {MODEL_ZIP}")

    _setup_cpu()
    actor, env_id = _build_actor(MODEL_ZIP)

    env = _make_env(env_id)
    act_space = env.action_space
    assert isinstance(act_space, spaces.Box)
    env.close()

    seeds = f"{START_SEED}..{START_SEED + N_SEEDS - 1}"
    print(f"Strategy: Fixed full SAC (b1_full_only)")
    print(f"Weight: {MODEL_ZIP}")
    print(f"environment: {env_id}")
    print(
        f"deadline∈[{DEADLINE_LOW_MS},{DEADLINE_HIGH_MS}] ms | every{DECISION_INTERVAL} step resampling | "
        f"miss→previous step | seeds {seeds}\n"
        f"Forward: tanh+unscale (not predict)\n"
        f"Success: Reach the end point without collision (any step info[cost]>0 is considered a collision)"
    )
    print(f"\nWarm-up (real environment, single core {CPU_CORE}):")
    _warmup(actor, env_id, act_space, START_SEED)

    results: list[EpisodeStats] = []
    print("\nFormal evaluation:")
    for seed in range(START_SEED, START_SEED + N_SEEDS):
        st = _run_episode(actor, env_id, act_space, seed)
        hit_rate = st.deadline_hits / max(st.n_steps, 1)
        results.append(st)
        print(
            f"  seed={seed} return={st.return_:.4f} steps={st.n_steps} "
            f"goal={st.goal_at_end} collision={st.had_collision} "
            f"hit_rate={hit_rate:.4f} success={st.success}"
        )

    rets = np.array([r.return_ for r in results], dtype=np.float64)
    steps = np.array([r.n_steps for r in results], dtype=np.float64)
    hit_rates = np.array(
        [r.deadline_hits / max(r.n_steps, 1) for r in results], dtype=np.float64
    )
    success_rate = float(np.mean([r.success for r in results]))
    collision_rate = float(np.mean([r.had_collision for r in results]))

    print(f"\n{'=' * 70}")
    print(f"comprehensive ({N_SEEDS} seeds) — fixed full SAC:")
    print(f"  mean return     = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  average step    = {steps.mean():.4f} ± {steps.std(ddof=0):.4f}")
    print(f"  mean hit rate   = {hit_rates.mean():.4f} ± {hit_rates.std(ddof=0):.4f}")
    print(f"  success rate    = {success_rate:.4f} ({sum(r.success for r in results)}/{N_SEEDS})")
    print(f"  collision rate  = {collision_rate:.4f} ({sum(r.had_collision for r in results)}/{N_SEEDS})")


if __name__ == "__main__":
    main()
