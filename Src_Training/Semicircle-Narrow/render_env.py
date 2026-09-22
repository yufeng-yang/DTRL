"""Semicircular track rendering: supports joint SAC ``.pt`` or SB3 SAC/PPO ``.zip``, or manual remote operation.

Project root directory execution:

Strategy rendering (default)::

    python render_env.py
    python render_env.py --model path/to/best_model.zip # SB3 full SAC
    python render_env.py --model path/to/best_model.pt # three-exit joint SAC

Manual keyboard::

    python render_env.py --teleop

Rendering and frame rate (environment variables, both set by this script before import mujoco / safety_gymnasium):
- ``MUJOCO_GL``: When not set, if there is ``DISPLAY``, the default is ``glfw``; if there is no display, it will be ``egl``.
- ``SEMICIRCLE_TELEOP_FPS``: main loop target frame rate, default **60**.
- ``SEMICIRCLE_MUJOCO_GL``: If set, overrides the default ``MUJOCO_GL`` above."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _configure_mujoco_gl() -> None:
    """The MuJoCo backend must be set up before importing mujoco for the first time."""
    if os.environ.get("MUJOCO_GL"):
        return
    override = os.environ.get("SEMICIRCLE_MUJOCO_GL")
    if override:
        os.environ["MUJOCO_GL"] = override
        return
    if os.environ.get("DISPLAY"):
        os.environ["MUJOCO_GL"] = "glfw"
    else:
        os.environ["MUJOCO_GL"] = "egl"


_configure_mujoco_gl()

import time
import tkinter as tk

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO, SAC

import Semicircle_env  # noqa: F401
import safety_gymnasium

_ROOT = Path(__file__).resolve().parent
_OFFPOLICY_DIR = _ROOT / "DTRL-Off"
if str(_OFFPOLICY_DIR) not in sys.path:
    sys.path.insert(0, str(_OFFPOLICY_DIR))

from action_utils import unscale_action  # noqa: E402
from eenn_network import ActorEENN3Semicircle, ActorEENN3SemicircleLarge  # noqa: E402

# SB3 full SAC (.zip); three-port joint please change to .pt, for example DTRL-Off/runs/.../best_model.pt
DEFAULT_MODEL = (
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/Full/runs/"
    "Full_20260512_115359/best_model/Full_best_model.zip"
)

ENV_ID = os.environ.get("SEMICIRCLE_ENV_ID", "SafetyPointSemicircle0-v5")
FPS = float(os.environ.get("SEMICIRCLE_TELEOP_FPS", "60"))
ACTION_SCALE = float(os.environ.get("SEMICIRCLE_TELEOP_SCALE", "0.8"))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(action, low, high).astype(np.float32)


class SafetyToGymnasiumWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["cost"] = float(cost)
        return obs, reward, terminated, truncated, info


def _make_render_env(env_id: str) -> gym.Env:
    env = safety_gymnasium.make(env_id, render_mode="human")
    return SafetyToGymnasiumWrapper(env)


def _load_sb3(model_path: Path, env: gym.Env):
    last: BaseException | None = None
    for loader in (SAC.load, PPO.load):
        try:
            return loader(str(model_path), env=env)
        except BaseException as e:
            last = e
    assert last is not None
    raise RuntimeError(f"Unable to load with PPO/SAC {model_path}") from last


def _actor_class_from_state(state: dict) -> type:
    w = state["exit1_mean.weight"]
    return ActorEENN3SemicircleLarge if int(w.shape[1]) == 128 else ActorEENN3Semicircle


def load_joint_sac_actor(
    model_path: Path,
    obs_dim: int,
    act_dim: int,
    device: torch.device,
) -> tuple[torch.nn.Module, str, dict]:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    actor_cls = _actor_class_from_state(ckpt["actor"])
    actor = actor_cls(obs_dim, act_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    env_id = str(ckpt.get("env_id", ENV_ID))
    return actor, env_id, ckpt


@torch.no_grad()
def policy_action(
    actor: torch.nn.Module,
    obs: np.ndarray,
    action_space: spaces.Box,
    device: torch.device,
    exit_id: int,
) -> np.ndarray:
    o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    m1, _ls1, m2, _ls2, m3, _ls3 = actor.forward(o)
    mean = m1 if exit_id == 1 else (m2 if exit_id == 2 else m3)
    a_scaled = torch.tanh(mean).cpu().numpy().reshape(-1)
    return unscale_action(a_scaled, action_space)


def run_teleop(env_id: str) -> None:
    _log(
        f"Semicircle_env manual mode env={env_id!r} MUJOCO_GL={os.environ.get('MUJOCO_GL', '')!r} "
        f"Target FPS={FPS:g}"
    )
    _log("Key position: W/S forward and backward, A/D left and right, Q/ESC exit")
    _log("Create environment...")
    env = safety_gymnasium.make(env_id, render_mode="human")
    _log("Environment created, resetting() (may be slow the first time)...")
    obs, _ = env.reset()

    pressed = {"w": False, "a": False, "s": False, "d": False}
    running = {"ok": True}

    root = tk.Tk()
    root.title(f"Semicircle Teleop ({env_id})")
    root.geometry("420x120")
    tk.Label(
        root,
        text="W/S before and after, A/D left and right, Q/ESC to exit; exit automatically after a single round",
    ).pack(pady=20)
    root.focus_force()

    def on_press(event) -> None:
        key = (event.keysym or "").lower()
        if key in pressed:
            pressed[key] = True
        elif key in ("q", "escape"):
            running["ok"] = False

    def on_release(event) -> None:
        key = (event.keysym or "").lower()
        if key in pressed:
            pressed[key] = False

    root.bind("<KeyPress>", on_press)
    root.bind("<KeyRelease>", on_release)

    frame_dt = 1.0 / max(FPS, 1.0)
    next_frame = time.perf_counter()
    while running["ok"]:
        root.update_idletasks()
        root.update()

        forward = float(pressed["w"]) - float(pressed["s"])
        turn = float(pressed["a"]) - float(pressed["d"])
        action = np.array([forward, turn], dtype=np.float32) * ACTION_SCALE
        action = _clip_action(action, env.action_space.low, env.action_space.high)

        obs, reward, cost, term, trunc, info = env.step(action)
        env.render()
        if term or trunc:
            _log(
                f"End of round: term={term}, trunc={trunc}, reward={reward:.3f}, cost={cost:.3f}"
            )
            break

        next_frame += frame_dt
        sleep_for = next_frame - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_frame = time.perf_counter()

    root.destroy()
    env.close()


def _run_episode_loop(env: gym.Env, act_fn, *, label: str, seed: int, episodes: int) -> None:
    frame_dt = 1.0 / max(FPS, 1.0)
    for ep in range(episodes):
        ep_seed = seed + ep
        obs, _ = env.reset(seed=ep_seed)
        ep_ret = 0.0
        ep_len = 0
        terminated = truncated = False
        next_frame = time.perf_counter()
        while not (terminated or truncated):
            action = act_fn(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            env.render()
            ep_ret += float(reward)
            ep_len += 1
            next_frame += frame_dt
            sleep_for = next_frame - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame = time.perf_counter()
        _log(
            f"{label} episode {ep + 1}/{episodes} (seed={ep_seed}): "
            f"return={ep_ret:.3f} len={ep_len} term={terminated} trunc={truncated}"
        )


def run_sb3_policy(
    model_path: Path,
    *,
    env_id: str | None = None,
    seed: int = 42,
    episodes: int = 2,
) -> None:
    use_env_id = env_id or ENV_ID
    env = _make_render_env(use_env_id)
    _log(f"Load SB3 model: {model_path}")
    model = _load_sb3(model_path, env)
    _log(f"Strategy rendering env={use_env_id!r}(Single full MLP) episodes={episodes} seed={seed}")

    def act_fn(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(-1)

    _run_episode_loop(env, act_fn, label="full_sac", seed=seed, episodes=episodes)
    env.close()


def run_joint_policy(
    model_path: Path,
    *,
    env_id: str | None = None,
    exit_ids: list[int] | None = None,
    seed: int = 42,
    episodes: int = 2,
    device: str = "cpu",
) -> None:
    dev = torch.device(device)
    _log(f"Load joint SAC: {model_path}")
    use_env_id = env_id or ENV_ID
    env = _make_render_env(use_env_id)
    assert isinstance(env.action_space, spaces.Box)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    actor, ckpt_env_id, ckpt = load_joint_sac_actor(model_path, obs_dim, act_dim, dev)
    if env_id is None:
        use_env_id = ckpt_env_id
        env.close()
        env = _make_render_env(use_env_id)

    exit_names = {1: "e1", 2: "e2", 3: "efull"}
    exits = exit_ids if exit_ids is not None else [1, 2, 3]
    _log(
        f"Strategy rendering env={use_env_id!r} exits={exits} "
        f"episodes_per_exit={episodes} seed={seed} device={dev}"
    )
    if "step" in ckpt:
        _log(f"checkpoint step={ckpt['step']}")
    for key in ("mean_return_e1", "mean_return_e2", "mean_return_efull", "weighted_mean_return"):
        if key in ckpt:
            _log(f"  {key}={ckpt[key]}")

    for exit_id in exits:
        exit_name = exit_names[exit_id]
        _log(f"--- exit {exit_id} ({exit_name}) ---")
        base_seed = seed + (exit_id - 1) * episodes

        def act_fn(obs: np.ndarray, eid: int = exit_id) -> np.ndarray:
            return policy_action(actor, obs, env.action_space, dev, eid)

        _run_episode_loop(env, act_fn, label=exit_name, seed=base_seed, episodes=episodes)

    env.close()


def run_policy(
    model_path: Path,
    *,
    env_id: str | None = None,
    exit_ids: list[int] | None = None,
    seed: int = 42,
    episodes: int = 2,
    device: str = "cpu",
) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_path.suffix.lower() == ".zip":
        if exit_ids is not None:
            _log("[WARN] SB3 .zip is single policy, --exit will be ignored")
        run_sb3_policy(model_path, env_id=env_id, seed=seed, episodes=episodes)
        return

    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Unsupported model formats: {model_path}(Please use .zip or .pt)")

    run_joint_policy(
        model_path,
        env_id=env_id,
        exit_ids=exit_ids,
        seed=seed,
        episodes=episodes,
        device=device,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semicircle rendering: default joint SAC, --teleop is manual")
    p.add_argument(
        "--teleop",
        action="store_true",
        help="Manual W/S/A/D control (joint SAC best_model loaded by default)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL),
        help="SB3 .zip or joint SAC .pt",
    )
    p.add_argument("--env-id", type=str, default=None, help="Override env_id in checkpoint")
    p.add_argument(
        "--exit",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Only render the specified exit; if omitted, e1/e2/efull will run each --episodes episode",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=2, help="Number of episodes per outlet")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.teleop:
        run_teleop(args.env_id or ENV_ID)
    else:
        run_policy(
            Path(args.model).expanduser().resolve(),
            env_id=args.env_id,
            exit_ids=[args.exit] if args.exit is not None else None,
            seed=args.seed,
            episodes=args.episodes,
            device=args.device,
        )


if __name__ == "__main__":
    main()
