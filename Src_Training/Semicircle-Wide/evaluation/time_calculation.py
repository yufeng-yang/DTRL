"""Decompose the e1/e2/e3 independent models (each containing a complete subnet) from the joint SAC weights, and then measure the inference time.

1. Export: evaluation/standalone_models/<joint_run>/e1.pt e2.pt e3.pt
2. Test: warm-up 1 episode, official 10 episodes, only output official statistics

Run: python time_calculation.py"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

_BOARD = Path(__file__).resolve().parent.parent
_DTRL = _BOARD / "DTRL-Off"
_EVAL = Path(__file__).resolve().parent
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
STANDALONE_DIR = _EVAL / "standalone_models" / "DTRL-Off_20260517_205942"
START_SEED = 42
N_EPISODES = 10
CPU_CORE = 2
_PCTS = (1, 5, 10, 90, 95, 97, 99)

# (tag, exit_id, structure description)
_EXITS: list[tuple[str, int, str]] = [
    ("e1", 1, "obs→256→out"),
    ("e2", 2, "obs→256→128→128→out"),
    ("e3", 3, "obs→256→128→128→64→64→out"),
]


class SafetyToGymnasiumWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["cost"] = float(cost)
        return obs, reward, terminated, truncated, info


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


_EXIT_CLS: dict[int, type[nn.Module]] = {1: Exit1, 2: Exit2, 3: Exit3}


def _setup_cpu() -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {CPU_CORE})
        print(f"[cpu] bind core {CPU_CORE}，affinity={sorted(os.sched_getaffinity(0))}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _assemble_from_joint(
    full: ActorEENN3Semicircle, exit_id: int, obs_dim: int, act_dim: int
) -> nn.Module:
    """Assemble independent subnets from joint actors (each port has its own backbone, no shared modules)."""
    net: nn.Module = _EXIT_CLS[exit_id](obs_dim, act_dim)
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


def export_standalone_models(
    joint_path: Path,
    out_dir: Path,
    env_id: str,
    obs_dim: int,
    act_dim: int,
) -> dict[str, Path]:
    """Export e1/e2/e3 as independent .pt (only containing the state_dict of this port subnet)."""
    ckpt = torch.load(joint_path, map_location="cpu", weights_only=False)
    full = ActorEENN3Semicircle(obs_dim, act_dim)
    full.load_state_dict(ckpt["actor"])
    full.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for tag, exit_id, arch in _EXITS:
        net = _assemble_from_joint(full, exit_id, obs_dim, act_dim)
        out_path = out_dir / f"{tag}.pt"
        torch.save(
            {
                "tag": tag,
                "exit_id": exit_id,
                "arch": arch,
                "env_id": env_id,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "joint_model": str(joint_path),
                "state_dict": net.state_dict(),
            },
            out_path,
        )
        paths[tag] = out_path
        print(f"  [export] {tag} → {out_path}")
    return paths


def load_standalone(model_path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    exit_id = int(ckpt["exit_id"])
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    net: nn.Module = _EXIT_CLS[exit_id](obs_dim, act_dim)
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt


def _infer(actor: nn.Module, obs: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    return torch.tanh(actor(x)).detach().numpy().flatten()


def _timed_step(
    actor: nn.Module, obs: np.ndarray, action_space: spaces.Box
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        action = unscale_action(_infer(actor, obs), action_space)
    return action, (time.perf_counter() - t0) * 1e3


def _warmup_episode(
    actor: nn.Module, env_id: str, action_space: spaces.Box, seed: int
) -> None:
    env = _make_env(env_id)
    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = _timed_step(actor, obs, action_space)
        obs, _, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
    env.close()


def _latency_stats(times: np.ndarray) -> dict[str, float]:
    d = {"mean": float(times.mean()), "median": float(np.median(times))}
    for p in _PCTS:
        d[f"p{p}"] = float(np.percentile(times, p))
    return d


def _print_stats(title: str, dist: dict[str, float], extra: str = "") -> None:
    suffix = f" | {extra}" if extra else ""
    print(f"\n=== {title}{suffix} ===")
    print(
        f"  p1={dist['p1']:.4f}  p5={dist['p5']:.4f}  p10={dist['p10']:.4f}  "
        f"mean={dist['mean']:.4f}  median={dist['median']:.4f}  "
        f"p90={dist['p90']:.4f}  p95={dist['p95']:.4f}  p97={dist['p97']:.4f}  "
        f"p99={dist['p99']:.4f}  (ms/step)"
    )


def _make_env(env_id: str) -> gym.Env:
    env = safety_gymnasium.make(env_id, render_mode=None)
    return SafetyToGymnasiumWrapper(env)


def _rollout_episodes(
    actor: nn.Module, env_id: str, action_space: spaces.Box, start_seed: int, n_episodes: int
) -> tuple[list[float], list[float]]:
    all_lat: list[float] = []
    returns: list[float] = []
    for ep in range(n_episodes):
        seed = start_seed + ep
        env = _make_env(env_id)
        obs, _ = env.reset(seed=seed)
        ep_ret = 0.0
        done = False
        while not done:
            action, dt_ms = _timed_step(actor, obs, action_space)
            all_lat.append(dt_ms)
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += float(r)
            done = bool(term or trunc)
        env.close()
        returns.append(ep_ret)
    return returns, all_lat


def main() -> None:
    if not JOINT_MODEL.is_file():
        raise SystemExit(f"Unable to find joint weights: {JOINT_MODEL}")

    _setup_cpu()
    joint_ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
    env_id = joint_ckpt.get("env_id", "SafetyPointSemicircle0-v6")

    env = _make_env(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_space = env.action_space
    assert isinstance(act_space, spaces.Box)
    act_dim = int(np.prod(act_space.shape))
    env.close()

    print(f"Joint weight: {JOINT_MODEL}")
    print(f"Independent model directory: {STANDALONE_DIR}\n")
    print("Step 1/2 — Export e1 / e2 / e3 independent models:")
    model_paths = export_standalone_models(
        JOINT_MODEL, STANDALONE_DIR, env_id, obs_dim, act_dim
    )

    seeds = f"{START_SEED}..{START_SEED + N_EPISODES - 1}"
    print(
        f"\nStep 2/2 — Load and test from standalone .pt | env_id={env_id}\n"
        f"Preheat=1 episode (seed={START_SEED}) | formal ={N_EPISODES} episodes (seeds {seeds}) | "
        f"Timing segment=tanh(mean)+unscale"
    )

    for tag, _, arch in _EXITS:
        path = model_paths[tag]
        actor, meta = load_standalone(path)
        assert meta["arch"] == arch

        print(f"\n{'=' * 70}")
        print(f"exit [{tag}] {arch}")
        print(f"Independent weight: {path}")

        _warmup_episode(actor, env_id, act_space, START_SEED)

        returns, all_lat = _rollout_episodes(
            actor, env_id, act_space, START_SEED, N_EPISODES
        )
        eval_times = np.asarray(all_lat, dtype=np.float64)
        mean_ret = float(np.mean(returns))
        _print_stats(
            f"{tag} official statistics ({N_EPISODES} episodes, standalone)",
            _latency_stats(eval_times),
            f"n_steps={len(eval_times)} mean_return={mean_ret:.4f}",
        )


if __name__ == "__main__":
    main()
