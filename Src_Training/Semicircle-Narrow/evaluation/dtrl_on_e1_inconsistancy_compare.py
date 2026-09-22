"""Semicircle narrow — Cross-exit inconsistency for DTRL_ON only: comparison of whether to include e1.

Two evaluations (remaining settings identical):
  A) Including e1: outlet e1 / e2 / ef (three outlets)
  B) Excluding e1: outlet e2 / ef (two outlets)

Run: python dtrl_on_e1_inconsistancy_compare.py
  (requires safety_gymnasium, such as conda activate sb3sg)"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO

_EVAL = Path(__file__).resolve().parent
_NARROW = _EVAL.parent
_DTRL = _NARROW / "DTRL-Off"
sys.path.insert(0, str(_NARROW))
sys.path.insert(0, str(_DTRL))

import Semicircle_env  # noqa: F401
import safety_gymnasium  # noqa: E402

ENV_ID = "SafetyPointSemicircle0-v5"
N_EPISODES = 10
START_SEED = 42
N_WORKERS = 8
OBS_ROUND_DECIMALS = 6
INFER_BATCH = 4096

DTRL_ON_PATHS: list[tuple[str, Path]] = [
    (
        "e1",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/DTRL-On/runs/"
            "DTRL-On_e1_20260516_050501/checkpoints/DTRL-On_e1_300000_steps.zip"
        ),
    ),
    (
        "e2",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/DTRL-On/runs/"
            "DTRL-On_e2_20260516_053958/best_model/DTRL-On_e2_best_model.zip"
        ),
    ),
    (
        "ef",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/DTRL-On/runs/"
            "DTRL-On_full_20260516_021458/best_model/DTRL-On_full_best_model.zip"
        ),
    ),
]

# Path index: 0=e1, 1=e2, 2=ef
ALL_EXIT_INDICES = [0, 1, 2]
WITHOUT_E1_EXIT_INDICES = [1, 2]


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
class ExitPolicy:
    tag: str
    actor: PpoActor

    def env_actions(self, obs_batch: np.ndarray, action_space: spaces.Box) -> np.ndarray:
        with torch.inference_mode():
            x = torch.as_tensor(obs_batch, dtype=torch.float32)
            raw = self.actor(x)
            low, high = action_space.low, action_space.high
            return np.clip(raw.detach().cpu().numpy(), low, high).astype(np.float32)


def _make_env(env_id: str) -> gym.Env:
    env = safety_gymnasium.make(env_id, render_mode=None)
    return SafetyToGymnasiumWrapper(env)


def _load_dtrl_on_policies(exit_indices: list[int]) -> list[ExitPolicy]:
    policies: list[ExitPolicy] = []
    for i in exit_indices:
        tag, path = DTRL_ON_PATHS[i]
        if not path.is_file():
            raise FileNotFoundError(path)
        model = PPO.load(str(path), device="cpu")
        actor = PpoActor(
            model.policy.mlp_extractor.policy_net,
            model.policy.action_net,
        ).eval()
        del model
        policies.append(ExitPolicy(tag, actor))
    return policies


def _obs_key(obs: np.ndarray) -> bytes:
    rounded = np.round(obs.astype(np.float64), OBS_ROUND_DECIMALS)
    return rounded.tobytes()


@dataclass(frozen=True)
class _RolloutJob:
    config_key: str
    slot: int
    exit_index: int
    seed: int
    obs_dim: int
    act_dim: int
    act_low: tuple[float, ...]
    act_high: tuple[float, ...]


def _box_from_job(job: _RolloutJob) -> spaces.Box:
    low = np.asarray(job.act_low, dtype=np.float32)
    high = np.asarray(job.act_high, dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


@lru_cache(maxsize=8)
def _policy_for_exit(config_key: str, exit_index: int) -> ExitPolicy:
    tag, path = DTRL_ON_PATHS[exit_index]
    if not path.is_file():
        raise FileNotFoundError(path)
    model = PPO.load(str(path), device="cpu")
    actor = PpoActor(
        model.policy.mlp_extractor.policy_net,
        model.policy.action_net,
    ).eval()
    del model
    return ExitPolicy(tag, actor)


def _rollout_one_episode(job: _RolloutJob) -> tuple[int, list[np.ndarray]]:
    torch.set_num_threads(1)
    policy = _policy_for_exit(job.config_key, job.exit_index)
    action_space = _box_from_job(job)
    states: list[np.ndarray] = []
    env = _make_env(ENV_ID)
    obs, _ = env.reset(seed=job.seed)
    done = False
    while not done:
        states.append(obs.copy())
        action = policy.env_actions(obs.reshape(1, -1), action_space)[0]
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
    env.close()
    return job.slot, states


def _parallel_rollout_states(
    config_key: str,
    exit_indices: list[int],
    seeds: list[int],
    action_space: spaces.Box,
    obs_dim: int,
    act_dim: int,
    *,
    n_workers: int = N_WORKERS,
) -> list[list[np.ndarray]]:
    jobs = [
        _RolloutJob(
            config_key=config_key,
            slot=slot,
            exit_index=exit_i,
            seed=seed,
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=tuple(float(x) for x in action_space.low.flatten()),
            act_high=tuple(float(x) for x in action_space.high.flatten()),
        )
        for slot, exit_i in enumerate(exit_indices)
        for seed in seeds
    ]
    per_exit: list[list[np.ndarray]] = [[] for _ in range(len(exit_indices))]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for slot, ep_states in pool.map(_rollout_one_episode, jobs):
            per_exit[slot].extend(ep_states)
    return per_exit


def _union_states(per_exit_states: list[list[np.ndarray]]) -> np.ndarray:
    seen: dict[bytes, np.ndarray] = {}
    for traj in per_exit_states:
        for obs in traj:
            key = _obs_key(obs)
            if key not in seen:
                seen[key] = obs.astype(np.float32)
    if not seen:
        raise ValueError("S_eval is empty")
    return np.stack(list(seen.values()), axis=0)


def _pairwise_inconsistency(actions: np.ndarray, action_denom: float) -> float:
    n = actions.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(actions[i] - actions[j])) / action_denom
            n_pairs += 1
    return total / n_pairs


def _batched_actions(
    policy: ExitPolicy,
    obs_batch: np.ndarray,
    action_space: spaces.Box,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, obs_batch.shape[0], INFER_BATCH):
        end = min(start + INFER_BATCH, obs_batch.shape[0])
        chunks.append(policy.env_actions(obs_batch[start:end], action_space))
    return np.concatenate(chunks, axis=0)


def compute_dtrl_on_inconsistency(
    label: str,
    exit_indices: list[int],
    action_space: spaces.Box,
    seeds: list[int],
    obs_dim: int,
    act_dim: int,
    *,
    n_workers: int = N_WORKERS,
) -> dict[str, float | int | str | list[str]]:
    config_key = f"{label}:{','.join(map(str, exit_indices))}"
    policies = _load_dtrl_on_policies(exit_indices)
    n = len(policies)
    exit_tags = [DTRL_ON_PATHS[i][0] for i in exit_indices]

    per_exit_states = _parallel_rollout_states(
        config_key, exit_indices, seeds, action_space, obs_dim, act_dim, n_workers=n_workers
    )
    s_eval = _union_states(per_exit_states)
    action_denom = float(np.linalg.norm(action_space.high - action_space.low))

    all_actions = [
        _batched_actions(policies[i], s_eval, action_space) for i in range(n)
    ]
    per_state = np.empty(s_eval.shape[0], dtype=np.float64)
    for t in range(s_eval.shape[0]):
        acts = np.stack([all_actions[i][t] for i in range(n)], axis=0)
        per_state[t] = _pairwise_inconsistency(acts, action_denom)

    return {
        "label": label,
        "exits": exit_tags,
        "n_exits": n,
        "n_episodes_per_exit": len(seeds),
        "states_per_exit": [len(s) for s in per_exit_states],
        "s_eval_size": int(s_eval.shape[0]),
        "action_denom_l2": action_denom,
        "inconsistency_mean": float(per_state.mean()),
        "inconsistency_std": float(per_state.std(ddof=0)),
        "inconsistency_p95": float(np.percentile(per_state, 95)),
    }


def main() -> None:
    for i in ALL_EXIT_INDICES:
        if not DTRL_ON_PATHS[i][1].is_file():
            raise SystemExit(f"Weight not found [{DTRL_ON_PATHS[i][0]}]: {DTRL_ON_PATHS[i][1]}")

    env = _make_env(ENV_ID)
    assert isinstance(env.action_space, spaces.Box)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_space = env.action_space
    env.close()

    seeds = list(range(START_SEED, START_SEED + N_EPISODES))
    configs = [
        ("Contains e1 (e1,e2,ef)", ALL_EXIT_INDICES),
        ("excluding e1 (e2,ef)", WITHOUT_E1_EXIT_INDICES),
    ]

    print(
        f"DTRL_ON Cross-exit inconsistency comparison | env={ENV_ID} | per export {N_EPISODES} episodes "
        f"(seeds {seeds[0]}..{seeds[-1]}) | {N_WORKERS} Process parallelism | No core binding"
    )
    print(
        f"Normalization: ||a_max-a_min||_2 = "
        f"{float(np.linalg.norm(action_space.high - action_space.low)):.6f}\n"
    )

    rows: list[dict[str, float | int | str | list[str]]] = []
    for label, exit_indices in configs:
        print(f"{'=' * 70}")
        print(f"Configuration: {label} | export={ [DTRL_ON_PATHS[i][0] for i in exit_indices] }")
        stats = compute_dtrl_on_inconsistency(
            label, exit_indices, action_space, seeds, obs_dim, act_dim
        )
        rows.append(stats)
        print(f"  Number of rollout states for each exit: {stats['states_per_exit']}")
        print(f"  |S_eval| = {stats['s_eval_size']}")
        print(f"  inconsistency (mean) = {stats['inconsistency_mean']:.6f}")
        print(f"  inconsistency (std)  = {stats['inconsistency_std']:.6f}")
        print(f"  inconsistency (p95)  = {stats['inconsistency_p95']:.6f}")

    print(f"\n{'=' * 70}")
    print("Comparison summary:")
    for r in rows:
        print(
            f"  {r['label']:22s}  exits={r['exits']}  "
            f"I={r['inconsistency_mean']:.6f}  |S_eval|={r['s_eval_size']}"
        )


if __name__ == "__main__":
    main()
