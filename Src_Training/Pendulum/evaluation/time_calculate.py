"""EENN joint weight → fixed ef subnet: distribution of single-step inference time consumption (ms/step) after binding core warm-up.

Run: python time_calculate.py"""

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

_DTRL = Path(__file__).resolve().parent.parent / "DTRL-Off"
sys.path.insert(0, str(_DTRL))

from action_utils import unscale_action  # noqa: E402
from eenn_network import ActorEENN3Deep, SharedBackbone  # noqa: E402

JOINT_MODEL = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/DTRL-Off/runs/"
    "three_exits_joint/DTRL-Off_Pendulum-v1_20260512_054426/best/DTRL-Off_best_model.pt"
)
SEED = 42
WARMUP_STEPS = 200
CPU_CORE = 2
_PCTS = (1, 5, 10, 90, 95, 97, 99)


class ExitEF(nn.Module):
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


def _timed_step(
    actor: nn.Module, obs: np.ndarray, action_space: spaces.Box
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        action = unscale_action(_infer(actor, obs), action_space)
    return action, (time.perf_counter() - t0) * 1e3


def _warmup(actor: nn.Module, obs_dim: int, action_space: spaces.Box) -> list[float]:
    z = np.zeros(obs_dim, dtype=np.float32)
    times: list[float] = []
    for _ in range(WARMUP_STEPS):
        _, lat_ms = _timed_step(actor, z, action_space)
        times.append(lat_ms)
    return times


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


def _rollout(
    actor: nn.Module, env_id: str, action_space: spaces.Box
) -> tuple[float, np.ndarray]:
    env = gym.make(env_id)
    obs, _ = env.reset(seed=SEED)
    ret, lat = 0.0, []
    done = False
    while not done:
        action, dt_ms = _timed_step(actor, obs, action_space)
        lat.append(dt_ms)
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r)
        done = bool(term or trunc)
    env.close()
    return ret, np.asarray(lat, dtype=np.float64)


def main() -> None:
    if not JOINT_MODEL.is_file():
        raise SystemExit(f"Not found: {JOINT_MODEL}")

    _setup_cpu()
    ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
    env_id = ckpt.get("env_id", "Pendulum-v1")
    env = gym.make(env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    act_space = env.action_space
    assert isinstance(act_space, spaces.Box)
    env.close()

    full = ActorEENN3Deep(obs_dim, act_dim)
    full.load_state_dict(ckpt["actor"])
    actor = _build_ef(full, obs_dim, act_dim)

    print(f"Joint weight: {JOINT_MODEL}")
    print(f"Exit: ef (exit3) | seed={SEED} | warmup={WARMUP_STEPS} | Timing segment=forward+unscale")

    warmup_times = np.asarray(_warmup(actor, obs_dim, act_space), dtype=np.float64)
    _print_stats("ef warm-up phase", _latency_stats(warmup_times), f"n={len(warmup_times)}")

    ep_ret, eval_times = _rollout(actor, env_id, act_space)
    _print_stats(
        "ef official statistics (seed=42 single game)",
        _latency_stats(eval_times),
        f"steps={len(eval_times)} return={ep_ret:.4f}",
    )


if __name__ == "__main__":
    main()
