"""Load Stable-Baselines3 weights (PPO or SAC), evaluate 100 consecutive episodes starting from seed=42, and print the average return.

Fill in the MODEL_ZIP absolute path in main() and run:
  pythondatashow.py

Note: onpolicy_e1_train.py saves PPO; b1_full_only saves SAC, which needs to be loaded using the corresponding algorithm."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC


def load_ppo_or_sac(model_path: Path, env: gym.Env):
    """Try PPO.load, SAC.load in order, matching checkpoint type."""
    last: BaseException | None = None
    for loader in (PPO.load, SAC.load):
        try:
            return loader(str(model_path), env=env)
        except BaseException as e:
            last = e
    assert last is not None
    raise RuntimeError(f"Unable to load with PPO/SAC {model_path}") from last

ENV_ID = "Pendulum-v1"
START_SEED = 42
NUM_SEEDS = 100


def main() -> None:
    MODEL_ZIP = (
        "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Pendulum/Full/runs/Full_20260512_044600/weights/Full_80000_steps.zip"
    )
    raw = Path(MODEL_ZIP).expanduser()
    if not raw.is_absolute():
        raise SystemExit(f"MODEL_ZIP must be an absolute path: {MODEL_ZIP}")
    model_path = raw.resolve()
    if not model_path.is_file():
        raise SystemExit(f"Weights file not found: {model_path}")

    env = gym.make(ENV_ID)
    model = load_ppo_or_sac(model_path, env)

    returns: list[float] = []
    for ep_seed in range(START_SEED, START_SEED + NUM_SEEDS):
        obs, _ = env.reset(seed=ep_seed)
        ep_ret = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
        returns.append(ep_ret)

    env.close()

    mean_ret = float(np.mean(returns))
    print(f"Loaded: {model_path}")
    print(f"environment: {ENV_ID}, seed range: [{START_SEED}, {START_SEED + NUM_SEEDS - 1}],common {NUM_SEEDS} bureau")
    print(f"average return: {mean_ret:.6f}")


if __name__ == "__main__":
    main()
