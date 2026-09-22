"""Cross-exit action inconsistency (pairwise normalized cross-exit action inconsistency).

For each method (MMS / DTRL_ON / DTRL_OFF):
  1. Each exit rolls out N_EPISODES independently and collects the access status set S^(i)
  2. S_eval = ⋃_i S^(i) (remove duplication according to rounding obs)
  3. For each s ∈ S_eval, three people reasoned with obs and got {a^(1),...,a^(N)}, and calculated the mean according to the formula

Run: python inconsistance_study.py"""

from __future__ import annotations

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
_DTRL = _EVAL.parent / "DTRL-Off"
sys.path.insert(0, str(_DTRL))

from action_utils import unscale_action  # noqa: E402
from eenn_network import ActorEENN3Deep, SharedBackbone  # noqa: E402

ENV_ID = "Pendulum-v1"
N_EXITS = 3
N_EPISODES = 10
START_SEED = 42
N_WORKERS = 8
OBS_ROUND_DECIMALS = 6
INFER_BATCH = 4096
PolicyKind = Literal["ppo", "sac", "eenn"]

JOINT_MODEL = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/DTRL-Off/runs/"
    "three_exits_joint/DTRL-Off_Pendulum-v1_20260512_054426/best/DTRL-Off_best_model.pt"
)

DTRL_ON_PATHS: list[tuple[str, Path, PolicyKind]] = [
    (
        "e1",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/DTRL-On/runs/"
            "DTRL-On_e1_sac128x128_20260512_080513/weights/best/DTRL-On_e1_best_model.zip"
        ),
        "ppo",
    ),
    (
        "e2",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/DTRL-On/runs/"
            "DTRL-On_e2_sac128x128x128_20260512_080918/weights/best/DTRL-On_e2_best_model.zip"
        ),
        "ppo",
    ),
    (
        "ef",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/Full/runs/"
            "Full_20260512_044600/weights/best/Full_best_model.zip"
        ),
        "sac",
    ),
]

MMS_PATHS: list[tuple[str, Path]] = [
    (
        "e1",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/SWI/runs/"
            "SWI_20260510_162815/weights/SWI_80000_steps.zip"
        ),
    ),
    (
        "e2",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/MMS/runs/"
            "MMS_20260512_050155/weights/MMS_80000_steps.zip"
        ),
    ),
    (
        "ef",
        Path(
            "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/Full/runs/"
            "Full_20260512_044600/weights/Full_80000_steps.zip"
        ),
    ),
]


# --- DTRL OFF (EENN joint) exit modules ---


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
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(64, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.trunk(self.fc3(self.fc2(self.backbone(obs)))))


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
        """obs (B, obs_dim) -> env-scale actions (B, act_dim)."""
        with torch.inference_mode():
            x = torch.as_tensor(obs_batch, dtype=torch.float32)
            raw = self.actor(x)
            if self.kind == "eenn":
                scaled_np = torch.tanh(raw).detach().cpu().numpy()
            elif self.kind == "sac":
                # SacActor.forward already contains tanh
                scaled_np = raw.detach().cpu().numpy()
            else:
                # PPO: Gaussian mean, clip to environment boundary
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

    def check_weights(self) -> None:
        return


def _build_eenn_exit(full: ActorEENN3Deep, exit_id: int, obs_dim: int, act_dim: int) -> nn.Module:
    row = next(x for x in _EENN_EXIT_SPECS if x[2] == exit_id)
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


class DtrlOffSuite(MethodSuite):
    name = "DTRL_OFF"

    def __init__(self, obs_dim: int, act_dim: int) -> None:
        if not JOINT_MODEL.is_file():
            raise FileNotFoundError(JOINT_MODEL)
        ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
        full = ActorEENN3Deep(obs_dim, act_dim)
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
        for tag, path, kind in DTRL_ON_PATHS:
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


class MmsSuite(MethodSuite):
    name = "MMS"

    def __init__(self) -> None:
        self._policies: list[ExitPolicy] = []
        for tag, path in MMS_PATHS:
            if not path.is_file():
                raise FileNotFoundError(path)
            model = SAC.load(str(path), device="cpu")
            actor = SacActor(model.policy.actor).eval()
            del model
            self._policies.append(ExitPolicy(tag, actor, "sac"))

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
        tag, path = MMS_PATHS[exit_index]
        model = SAC.load(str(path), device="cpu")
        actor = SacActor(model.policy.actor).eval()
        del model
        return ExitPolicy(tag, actor, "sac")
    if method == "DTRL_ON":
        tag, path, kind = DTRL_ON_PATHS[exit_index]
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
    if method == "DTRL_OFF":
        if not JOINT_MODEL.is_file():
            raise FileNotFoundError(JOINT_MODEL)
        ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
        full = ActorEENN3Deep(obs_dim, act_dim)
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
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=job.seed)
    done = False
    while not done:
        states.append(obs.copy())
        action = policy.env_actions(obs.reshape(1, -1), action_space)[0]
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
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
    """S_eval = ⋃_i S^(i), press rounded obs to remove duplicates, return (|S_eval|, obs_dim)."""
    seen: dict[bytes, np.ndarray] = {}
    for traj in per_exit_states:
        for obs in traj:
            key = _obs_key(obs)
            if key not in seen:
                seen[key] = obs.astype(np.float32)
    if not seen:
        raise ValueError("S_eval is empty")
    return np.stack(list(seen.values()), axis=0)


def _pairwise_inconsistency(
    actions: np.ndarray,
    action_denom: float,
) -> float:
    """actions: (N, act_dim) Environment-scale actions for each outlet on the same obs.    """
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
) -> dict[str, float | int]:
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
    env = gym.make(ENV_ID)
    assert isinstance(env.action_space, spaces.Box)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_space = env.action_space
    env.close()

    seeds = list(range(START_SEED, START_SEED + N_EPISODES))
    suites: list[MethodSuite] = [
        MmsSuite(),
        DtrlOnSuite(),
        DtrlOffSuite(obs_dim, act_dim),
    ]

    print(
        f"Action inconsistency across exits | env={ENV_ID} | per export {N_EPISODES} episodes "
        f"(seeds {seeds[0]}..{seeds[-1]}) | rollout {N_WORKERS} Process parallelism | No core binding"
    )
    print("|S_eval| is the union of three rollout states (remove duplication)")
    print(
        f"Normalization: ||a_max-a_min||_2 = {float(np.linalg.norm(action_space.high - action_space.low)):.6f}\n"
    )

    rows: list[dict[str, float | int]] = []
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
