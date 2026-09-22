"""Reproduce Table 3: exit returns under different loss-weight allocations.

Loads standalone E1/E2/E3 from ``checkpoints/Ablation/Ablation2``.
Does not retrain. Environment: SafetyPointSemicircle0-v6 (Semicircle-Wide).
"""

from envpy import ensure_gym_python

ensure_gym_python()

import csv
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Src_Training/Semicircle-Wide"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "DTRL-Off"))

import Semicircle_env
import safety_gymnasium
from action_utils import unscale_action
from eenn_network import SharedBackbone256

ENV_ID = "SafetyPointSemicircle0-v6"
N_EPISODES = 100             # number of evaluation episodes per exit and row
START_SEED = 42              # first episode seed (then +1, +2, …)
WEIGHT_ROOT = ROOT / "checkpoints/Ablation/Ablation2"
OUT_CSV = ROOT / "figure_and_table/Table 3 Ablation2.csv"
CONFIGS = (
    ("2 : 3 : 5", WEIGHT_ROOT / "2-3-5"),
    ("5 : 3 : 2", WEIGHT_ROOT / "5-3-2"),
)


class SafetyToGymnasiumWrapper(gym.Wrapper):
    """Expose Safety-Gymnasium's extra cost as ``info['cost']`` for Gymnasium."""
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info or {})
        info["cost"] = float(cost)
        return obs, reward, terminated, truncated, info


class Exit1(nn.Module):
    """Standalone Exit 1: obs → 256 → action mean."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.backbone = SharedBackbone256(obs_dim)
        self.mean = nn.Linear(self.backbone.out_dim, act_dim)

    def forward(self, obs):
        return self.mean(self.backbone(obs))


class Exit2(nn.Module):
    """Standalone Exit 2: obs → 256 → 128 → 128 → action mean."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.backbone = SharedBackbone256(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.mean = nn.Linear(128, act_dim)

    def forward(self, obs):
        return self.mean(self.fc3(self.fc2(self.backbone(obs))))


class Exit3(nn.Module):
    """Standalone Exit 3 / efull: obs → 256 → 128 → 128 → 64 → 64 → action mean."""
    def __init__(self, obs_dim, act_dim):
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

    def forward(self, obs):
        return self.mean(self.trunk(self.fc3(self.fc2(self.backbone(obs)))))


def load_exit(path):
    """Build the exit network named in ``path`` and load its ``state_dict``."""
    # Ablation checkpoints store each exit as a standalone state dictionary.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = {1: Exit1, 2: Exit2, 3: Exit3}[ckpt["exit_id"]](ckpt["obs_dim"], ckpt["act_dim"])
    net.load_state_dict(ckpt["state_dict"])
    return net.eval()


def eval_exit(actor, label, tag):
    """Mean undiscounted return of ``actor`` over ``N_EPISODES`` seeds starting at 42."""
    env = SafetyToGymnasiumWrapper(safety_gymnasium.make(ENV_ID, render_mode=None))
    rets = []
    for i in range(N_EPISODES):
        obs, _ = env.reset(seed=START_SEED + i)
        ep_ret = 0.0
        done = False
        while not done:
            with torch.inference_mode():
                x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                a = torch.tanh(actor(x)).cpu().numpy().reshape(-1)
            obs, reward, terminated, truncated, _ = env.step(unscale_action(a, env.action_space))
            ep_ret += float(reward)
            done = terminated or truncated
        rets.append(ep_ret)
        print(f"\r[{label}] {tag} {i + 1}/{N_EPISODES}", end="", flush=True)
    print()
    env.close()
    return float(np.mean(rets))


def main():
    """Evaluate both loss-weight folders and write ``Table 3 Ablation2.csv``."""
    rows = []
    # Evaluate every exit on the same 100 episode seeds.
    for label, folder in CONFIGS:
        e1 = eval_exit(load_exit(folder / "e1.pt"), label, "e1")
        e2 = eval_exit(load_exit(folder / "e2.pt"), label, "e2")
        e3 = eval_exit(load_exit(folder / "e3.pt"), label, "e3")
        rows.append((label, e1, e2, e3))

    header = f"{'Loss Weights (E1 : E2 : E3)':28s} {'E1':>8s} {'E2':>8s} {'E3':>8s}"
    print()
    print(header)
    print("-" * len(header))
    for label, e1, e2, e3 in rows:
        print(f"{label:28s} {e1:8.2f} {e2:8.2f} {e3:8.2f}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Loss Weights (E1 : E2 : E3)", "E1", "E2", "E3"])
        for label, e1, e2, e3 in rows:
            w.writerow([label, f"{e1:.2f}", f"{e2:.2f}", f"{e3:.2f}"])
    print(f"\nCSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
