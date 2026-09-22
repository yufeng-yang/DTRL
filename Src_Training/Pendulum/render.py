"""Load Stable-Baselines3 SAC weights and visualize the rollout using the Gymnasium environment.

Fill in the absolute path of MODEL_ZIP in main() and run:
  python render.py

The default is Pendulum-v1, and 3 episode seeds are randomly selected to play one round each; at the end, the rewards for each round and the seeds used are printed."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="Pendulum-v1")
    p.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of rendering sessions (default 3)",
    )
    p.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="RNG seed used to generate episode seeds for each game; if not set, it is non-deterministic",
    )
    return p.parse_args()


def main() -> None:
    # Change to the absolute path of your weights .zip (for example: .../weights/best/best_model.zip or .../weights/Full_final.zip)
    MODEL_ZIP = (
        "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/SWI/runs/SWI_20260510_162815/weights/Full_final.zip"
    )
    raw = Path(MODEL_ZIP).expanduser()
    if not raw.is_absolute():
        raise SystemExit(f"MODEL_ZIP must be an absolute path: {MODEL_ZIP}")
    model_path = raw.resolve()

    args = parse_args()
    if not model_path.is_file():
        raise SystemExit(f"Weights file not found: {model_path}")

    rng = np.random.default_rng(args.rng_seed)
    episode_seeds = [int(rng.integers(0, 2**31)) for _ in range(args.episodes)]

    env = gym.make(args.env_id, render_mode="human")
    model = SAC.load(str(model_path), env=env)

    print(f"Loaded: {model_path}")
    print(f"This batch {args.episodes} Bureau's episode seed: {episode_seeds}")

    for i, ep_seed in enumerate(episode_seeds, start=1):
        obs, _ = env.reset(seed=ep_seed)
        ep_ret = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
        print(f"No. {i} bureau (seed={ep_seed}) Cumulative return: {ep_ret:.4f}")

    env.close()


if __name__ == "__main__":
    main()
