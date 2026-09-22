"""Loading PPO/SAC weights without rendering evaluation on Semicircle v5.

Fill in the MODEL_ZIP absolute path in main() at the end of the file and run:
  python data_show.py"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC

_SEMICIRCLE_NARROW = Path(__file__).resolve().parent
if str(_SEMICIRCLE_NARROW) not in sys.path:
    sys.path.insert(0, str(_SEMICIRCLE_NARROW))

import Semicircle_env  # noqa: F401
import safety_gymnasium


class SafetyToGymnasiumWrapper(gym.Wrapper):
    """Safety-Gymnasium step API → Gymnasium (consistent with train_sac_full)."""

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["cost"] = float(cost)
        return obs, reward, terminated, truncated, info


def make_env(env_id: str) -> gym.Env:
    env = safety_gymnasium.make(env_id, render_mode=None)
    return SafetyToGymnasiumWrapper(env)


def load_ppo_or_sac(model_path: Path, env: gym.Env):
    last: BaseException | None = None
    for loader in (PPO.load, SAC.load):
        try:
            return loader(str(model_path), env=env)
        except BaseException as e:
            last = e
    assert last is not None
    raise RuntimeError(f"Unable to load with PPO/SAC {model_path}") from last


def evaluate(
    model_path: Path,
    env_id: str,
    num_episodes: int,
    start_seed: int,
) -> None:
    env = make_env(env_id)
    model = load_ppo_or_sac(model_path, env)

    rewards: list[float] = []
    steps_list: list[int] = []
    hits: list[bool] = []
    collision_counts: list[int] = []

    for i in range(num_episodes):
        seed = start_seed + i
        obs, _ = env.reset(seed=seed)
        ep_reward = 0.0
        ep_steps = 0
        ep_collisions = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            if info.get("cost", 0.0) > 0:
                ep_collisions += 1
        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        hits.append(ep_collisions > 0)
        collision_counts.append(ep_collisions)

    env.close()

    hit_rate = float(np.mean(hits))
    mean_reward = float(np.mean(rewards))
    mean_steps = float(np.mean(steps_list))
    mean_collisions = float(np.mean(collision_counts))

    print(f"Loaded: {model_path}")
    print(f"environment: {env_id}, no rendering, total {num_episodes} seed {start_seed}..{start_seed + num_episodes - 1}）")
    print(f"Impact rate (proportion of rounds with cost>0 for any step): {hit_rate * 100:.1f}% ({int(np.sum(hits))}/{num_episodes})")
    print(f"Average number of collisions (number of steps with cost>0 per round): {mean_collisions:.2f}")
    print(f"Number of collisions in each round: {collision_counts}")
    print(f"Average number of steps: {mean_steps:.2f}")
    print(f"Average return: {mean_reward:.4f}")


def main() -> None:
    # ----- The following paths and parameters can be replaced as needed (it is recommended to keep the absolute path) -----
    MODEL_ZIP = (
"/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/SWI/runs/SWI_20260516_000551/best_model/SWI_best_model.zip"
    )
    ENV_ID = "SafetyPointSemicircle0-v5"
    NUM_EPISODES = 10
    START_SEED = 42
    # -------------------------------------------------------

    raw = Path(MODEL_ZIP).expanduser()
    if not raw.is_absolute():
        raise SystemExit(f"MODEL_ZIP must be an absolute path: {MODEL_ZIP}")
    model_path = raw.resolve()
    if not model_path.is_file():
        raise SystemExit(f"Weights file not found: {model_path}")

    evaluate(
        model_path=model_path,
        env_id=ENV_ID,
        num_episodes=NUM_EPISODES,
        start_seed=START_SEED,
    )


if __name__ == "__main__":
    main()
