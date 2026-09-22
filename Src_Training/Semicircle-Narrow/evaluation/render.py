"""Render a round of PPO E1 (same weight as self_eval.py).

Run (requires DISPLAY / graphics environment):
  python render.py
  python render.py --seed 42"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _configure_mujoco_gl() -> None:
    if os.environ.get("MUJOCO_GL"):
        return
    override = os.environ.get("SEMICIRCLE_MUJOCO_GL")
    if override:
        os.environ["MUJOCO_GL"] = override
        return
    os.environ["MUJOCO_GL"] = "glfw" if os.environ.get("DISPLAY") else "egl"


_configure_mujoco_gl()

import argparse

import numpy as np
from stable_baselines3 import PPO

_NARROW = Path(__file__).resolve().parent.parent
_ONPOLICY = _NARROW / "DTRL-On"
sys.path.insert(0, str(_NARROW))
sys.path.insert(0, str(_ONPOLICY))

import Semicircle_env  # noqa: F401
import safety_gymnasium  # noqa: E402

from train_DTRL_On_full import SafetyToGymnasiumWrapper  # noqa: E402

MODEL_ZIP = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/DTRL-On/runs/"
    "DTRL-On_e1_20260516_050501/checkpoints/DTRL-On_e1_300000_steps.zip"
)
ENV_ID = "SafetyPointSemicircle0-v5"
DEFAULT_SEED = 42
FPS = float(os.environ.get("SEMICIRCLE_RENDER_FPS", "60"))


def _make_env(env_id: str):
    env = safety_gymnasium.make(env_id, render_mode="human")
    return SafetyToGymnasiumWrapper(env)


def _env_id_from_model(model: PPO, fallback: str) -> str:
    if model.env is not None and hasattr(model.env, "spec") and model.env.spec is not None:
        return str(model.env.spec.id)
    return fallback


def render_episode(
    model_path: Path,
    *,
    env_id: str = ENV_ID,
    seed: int = DEFAULT_SEED,
) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    env = _make_env(env_id)
    print(f"load: {model_path}")
    model = PPO.load(str(model_path), env=env, device="cpu")
    use_env_id = _env_id_from_model(model, env_id)
    if use_env_id != env_id:
        env.close()
        env = _make_env(use_env_id)

    frame_dt = 1.0 / max(FPS, 1.0)
    obs, _ = env.reset(seed=seed)
    ep_ret = 0.0
    ep_len = 0
    collision_steps = 0
    done = False
    next_frame = time.perf_counter()

    print(f"render env={use_env_id} seed={seed} fps={FPS}")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        ep_ret += float(reward)
        ep_len += 1
        if float(info.get("cost", 0.0)) > 0:
            collision_steps += 1

        done = bool(terminated or truncated)
        next_frame += frame_dt
        sleep_for = next_frame - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_frame = time.perf_counter()

    env.close()
    print(
        f"episode done | return={ep_ret:.4f} steps={ep_len} "
        f"collision_steps={collision_steps} term={terminated} trunc={truncated}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Rendering PPO E1 one game")
    p.add_argument("--model", type=Path, default=MODEL_ZIP, help="SB3 PPO .zip path")
    p.add_argument("--env-id", type=str, default=ENV_ID)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()
    render_episode(args.model, env_id=args.env_id, seed=args.seed)


if __name__ == "__main__":
    main()
