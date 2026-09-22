"""Semicircle narrow (v5) Action inconsistency across exits.

For MMS/DTRL_ON/DTRL_OFF:
  1. Each exit independently rolls out each N_EPISODES station and collects S^(i)
  2. S_eval = ⋃_i S^(i) (obs rounding and deduplication)
  3. Use the same reasoning as above obs and report inconsistency mean ± std (state by state)

Run: python inconsistancy_study.py
  (requires safety_gymnasium environment, such as conda activate sb3sg)"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, SAC

_EVAL = Path(__file__).resolve().parent
_NARROW = _EVAL.parent
_DTRL = _NARROW / "DTRL-Off"
sys.path.insert(0, str(_NARROW))
sys.path.insert(0, str(_DTRL))

import Semicircle_env  # noqa: F401
import safety_gymnasium  # noqa: E402

from action_utils import unscale_action  # noqa: E402
from eenn_network import ActorEENN3Semicircle, SharedBackbone256  # noqa: E402

ENV_ID = "SafetyPointSemicircle0-v5"
N_EXITS = 3
N_EPISODES = 10
START_SEED = 42
N_WORKERS = 8
OBS_ROUND_DECIMALS = 6
INFER_BATCH = 4096
PolicyKind = Literal["ppo", "sac", "eenn"]

JOINT_MODEL = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/Src_Training/Semicircle-Narrow/DTRL-Off/runs/"
    "DTRL-Off_20260516_034629/best_model/DTRL-Off_best_model.pt"
)

DTRL_ON_PATHS: list[tuple[str, Path]] = [
    (
        "e1",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/Src_Training/Semicircle-Narrow/DTRL-On/runs/"
            "DTRL-On_e1_20260516_050501/best_model/DTRL-On_e1_best_model.zip"
        ),
    ),
    (
        "e2",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/Src_Training/Semicircle-Narrow/DTRL-On/runs/"
            "DTRL-On_e2_20260516_053958/best_model/DTRL-On_e2_best_model.zip"
        ),
    ),
    (
        "ef",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/Src_Training/Semicircle-Narrow/DTRL-On/runs/"
            "DTRL-On_full_20260516_021458/best_model/DTRL-On_full_best_model.zip"
        ),
    ),
]

MMS_PATHS: list[tuple[str, Path, PolicyKind]] = [
    (
        "e1",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/Src_Training/Semicircle-Narrow/SWI/runs/"
            "SWI_20260516_000551/checkpoints/SWI_1500000_steps.zip"
        ),
        "ppo",
    ),
    (
        "e2",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/Src_Training/Semicircle-Narrow/MMS/runs/"
            "MMS_20260516_012652/checkpoints/MMS_500000_steps.zip"
        ),
        "ppo",
    ),
    (
        "ef",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/Src_Training/Semicircle-Narrow/Full/runs/"
            "Full_20260512_115359/best_model/Full_best_model.zip"
        ),
        "sac",
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


class Exit1(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone256(obs_dim)
        self.mean = nn.Linear(self.backbone.out_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.backbone(obs))


class Exit2(nn.Module):
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


_EENN_EXIT_SPECS: list[tuple[str, type[nn.Module], int]] = [
    ("e1", Exit1, 1),
    ("e2", Exit2, 2),
    ("ef", Exit3, 3),
]


class PpoActor(nn.Module):
    def __init__(self, policy_net: nn.Sequential, action_net: nn.Linear) -> None:
        super().__init__()
        self.trunk = policy_net
        self.head = action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))


class SacActor(nn.Module):
    def __init__(self, sac_actor: nn.Module) -> None:
        super().__init__()
        self.trunk = sac_actor.latent_pi
        self.head = sac_actor.mu

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.trunk(obs)))


@dataclass
class ExitPolicy:
    tag: str
    actor: nn.Module
    kind: PolicyKind

    def env_actions(self, obs_batch: np.ndarray, action_space: spaces.Box) -> np.ndarray:
        with torch.inference_mode():
            x = torch.as_tensor(obs_batch, dtype=torch.float32)
            raw = self.actor(x)
            if self.kind == "eenn":
                scaled_np = torch.tanh(raw).detach().cpu().numpy()
            elif self.kind == "sac":
                scaled_np = raw.detach().cpu().numpy()
            else:
                low, high = action_space.low, action_space.high
                return np.clip(raw.detach().cpu().numpy(), low, high).astype(np.float32)
            out = np.empty_like(scaled_np, dtype=np.float32)
            for i in range(scaled_np.shape[0]):
                out[i] = unscale_action(scaled_np[i], action_space)
            return out


class MethodSuite(ABC):
    name: str

    @abstractmethod
    def policies(self) -> list[ExitPolicy]:
        ...


def _make_env(env_id: str) -> gym.Env:
    env = safety_gymnasium.make(env_id, render_mode=None)
    return SafetyToGymnasiumWrapper(env)


def _build_eenn_exit(full: ActorEENN3Semicircle, exit_id: int, obs_dim: int, act_dim: int) -> nn.Module:
    row = next(x for x in _EENN_EXIT_SPECS if x[2] == exit_id)
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


class DtrlOffSuite(MethodSuite):
    name = "DTRL_OFF"

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        if not JOINT_MODEL.is_file():
            raise FileNotFoundError(JOINT_MODEL)
        ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
        full = ActorEENN3Semicircle(obs_dim, act_dim)
        full.load_state_dict(ckpt["actor"])
        full.eval()
        self._policies = [
            ExitPolicy(tag, _build_eenn_exit(full, eid, obs_dim, act_dim), "eenn")
            for tag, _, eid in _EENN_EXIT_SPECS
        ]

    def policies(self) -> list[ExitPolicy]:
        return self._policies


class DtrlOnSuite(MethodSuite):
    name = "DTRL_ON"

    def __init__(self) -> None:
        self._policies: list[ExitPolicy] = []
        for tag, path in DTRL_ON_PATHS:
            if not path.is_file():
                raise FileNotFoundError(path)
            model = PPO.load(str(path), device="cpu")
            actor = PpoActor(
                model.policy.mlp_extractor.policy_net,
                model.policy.action_net,
            ).eval()
            del model
            self._policies.append(ExitPolicy(tag, actor, "ppo"))

    def policies(self) -> list[ExitPolicy]:
        return self._policies


class MmsSuite(MethodSuite):
    name = "MMS"

    def __init__(self) -> None:
        self._policies: list[ExitPolicy] = []
        for tag, path, kind in MMS_PATHS:
            if not path.is_file():
                raise FileNotFoundError(path)
            if kind == "ppo":
                model = PPO.load(str(path), device="cpu")
                actor = PpoActor(
                    model.policy.mlp_extractor.policy_net,
                    model.policy.action_net,
                ).eval()
                del model
            else:
                model = SAC.load(str(path), device="cpu")
                actor = SacActor(model.policy.actor).eval()
                del model
            self._policies.append(ExitPolicy(tag, actor, kind))

    def policies(self) -> list[ExitPolicy]:
        return self._policies


def _obs_key(obs: np.ndarray) -> bytes:
    rounded = np.round(obs.astype(np.float64), OBS_ROUND_DECIMALS)
    return rounded.tobytes()


@dataclass(frozen=True)
class _RolloutJob:
    method: str
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


@lru_cache(maxsize=16)
def _policy_for_exit(method: str, exit_index: int, obs_dim: int, act_dim: int) -> ExitPolicy:
    if method == "MMS":
        tag, path, kind = MMS_PATHS[exit_index]
        if kind == "ppo":
            model = PPO.load(str(path), device="cpu")
            actor = PpoActor(
                model.policy.mlp_extractor.policy_net,
                model.policy.action_net,
            ).eval()
            del model
            return ExitPolicy(tag, actor, "ppo")
        model = SAC.load(str(path), device="cpu")
        actor = SacActor(model.policy.actor).eval()
        del model
        return ExitPolicy(tag, actor, "sac")
    if method == "DTRL_ON":
        tag, path = DTRL_ON_PATHS[exit_index]
        model = PPO.load(str(path), device="cpu")
        actor = PpoActor(
            model.policy.mlp_extractor.policy_net,
            model.policy.action_net,
        ).eval()
        del model
        return ExitPolicy(tag, actor, "ppo")
    if method == "DTRL_OFF":
        if not JOINT_MODEL.is_file():
            raise FileNotFoundError(JOINT_MODEL)
        ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
        full = ActorEENN3Semicircle(obs_dim, act_dim)
        full.load_state_dict(ckpt["actor"])
        full.eval()
        tag, _, eid = _EENN_EXIT_SPECS[exit_index]
        return ExitPolicy(tag, _build_eenn_exit(full, eid, obs_dim, act_dim), "eenn")
    raise ValueError(f"Unknown method: {method}")


def _rollout_one_episode(job: _RolloutJob) -> tuple[int, list[np.ndarray]]:
    torch.set_num_threads(1)
    policy = _policy_for_exit(job.method, job.exit_index, job.obs_dim, job.act_dim)
    action_space = _box_from_job(job)
    states: list[np.ndarray] = []
    env = _make_env(ENV_ID)
    try:
        obs, _ = env.reset(seed=job.seed)
        done = False
        while not done:
            states.append(obs.copy())
            action = policy.env_actions(obs.reshape(1, -1), action_space)[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
    finally:
        env.close()
    return job.exit_index, states


def _parallel_rollout_states(
    method: str,
    seeds: list[int],
    action_space: spaces.Box,
    obs_dim: int,
    act_dim: int,
    *,
    n_workers: int = N_WORKERS,
) -> list[list[np.ndarray]]:
    jobs = [
        _RolloutJob(
            method=method,
            exit_index=i,
            seed=seed,
            obs_dim=obs_dim,
            act_dim=act_dim,
            act_low=tuple(float(x) for x in action_space.low.flatten()),
            act_high=tuple(float(x) for x in action_space.high.flatten()),
        )
        for i in range(N_EXITS)
        for seed in seeds
    ]
    per_exit: list[list[np.ndarray]] = [[] for _ in range(N_EXITS)]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for exit_i, ep_states in pool.map(_rollout_one_episode, jobs):
            per_exit[exit_i].extend(ep_states)
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
    if action_denom <= 0:
        raise ValueError("action_denom must be positive")
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
    n = obs_batch.shape[0]
    chunks: list[np.ndarray] = []
    for start in range(0, n, INFER_BATCH):
        end = min(start + INFER_BATCH, n)
        chunks.append(policy.env_actions(obs_batch[start:end], action_space))
    return np.concatenate(chunks, axis=0)


def compute_method_inconsistency(
    suite: MethodSuite,
    action_space: spaces.Box,
    seeds: list[int],
    obs_dim: int,
    act_dim: int,
    *,
    n_workers: int = N_WORKERS,
) -> dict[str, float | int | str]:
    policies = suite.policies()
    n = len(policies)
    assert n == N_EXITS

    per_exit_states = _parallel_rollout_states(
        suite.name, seeds, action_space, obs_dim, act_dim, n_workers=n_workers
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
        "method": suite.name,
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
    parser = argparse.ArgumentParser(description="Semicircle narrow cross-exit inconsistency")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=N_EPISODES,
        help=f"The number of episodes for independent rollout per outlet (default {N_EPISODES}）",
    )
    parser.add_argument("--start-seed", type=int, default=START_SEED)
    args = parser.parse_args()
    n_episodes = max(1, int(args.n_episodes))
    seeds = list(range(args.start_seed, args.start_seed + n_episodes))

    env = _make_env(ENV_ID)
    assert isinstance(env.action_space, spaces.Box)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_space = env.action_space
    env.close()

    suites: list[MethodSuite] = [
        MmsSuite(),
        DtrlOnSuite(),
        DtrlOffSuite(obs_dim, act_dim),
    ]

    print(
        f"Action inconsistency across exits | env={ENV_ID} | per export {n_episodes} episode(s) "
        f"(seeds {seeds[0]}..{seeds[-1]}) | rollout {N_WORKERS} Process parallelism | No core binding"
    )
    print("|S_eval| is the union of three rollout states (remove duplication)")
    print(
        f"Normalization: ||a_max-a_min||_2 = "
        f"{float(np.linalg.norm(action_space.high - action_space.low)):.6f}\n"
    )

    rows: list[dict[str, float | int | str]] = []
    for suite in suites:
        print(f"{'=' * 70}")
        print(f"method: {suite.name}")
        stats = compute_method_inconsistency(
            suite, action_space, seeds, obs_dim, act_dim, n_workers=N_WORKERS
        )
        rows.append(stats)
        print(f"  Number of rollout states for each exit: {stats['states_per_exit']}")
        print(f"  |S_eval| = {stats['s_eval_size']}")
        print(
            f"  inconsistency       = {stats['inconsistency_mean']:.6f} "
            f"± {stats['inconsistency_std']:.6f}"
        )
        print(f"  inconsistency (p95) = {stats['inconsistency_p95']:.6f}")

    print(f"\n{'=' * 70}")
    print("Summary (in ascending order by mean; std is the standard deviation of state-by-state inconsistency on S_eval):")
    rows.sort(key=lambda r: float(r["inconsistency_mean"]))
    for r in rows:
        print(
            f"  {r['method']:10s}  I={r['inconsistency_mean']:.6f} "
            f"± {r['inconsistency_std']:.6f}  |S_eval|={r['s_eval_size']}"
        )


if __name__ == "__main__":
    main()
