"""Semicircle wide track (v6, lane half width 0.7) environment preview: MuJoCo window + keyboard driving.

Execute in the ``Semicircle-Wide`` directory::

    python env_show.py
    python env_show.py --robotCar

Key positions: W/S forward and backward, A/D left and right, R to restart a game, Q/ESC to exit.

Environment variables (must take effect before import mujoco):
- ``SEMICIRCLE_MUJOCO_GL`` / ``MUJOCO_GL``: MuJoCo backend (default glfw for monitors)
- ``SEMICIRCLE_TELEOP_FPS``: target frame rate, default 60
- ``SEMICIRCLE_TELEOP_SCALE``: action range, default 0.8"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _configure_mujoco_gl() -> None:
    if os.environ.get("MUJOCO_GL"):
        return
    override = os.environ.get("SEMICIRCLE_MUJOCO_GL")
    if override:
        os.environ["MUJOCO_GL"] = override
        return
    os.environ["MUJOCO_GL"] = "glfw" if os.environ.get("DISPLAY") else "egl"


_configure_mujoco_gl()

import Semicircle_env  # noqa: F401
import safety_gymnasium

# v6: wide track (LANE_HALF_WIDTH = 0.7, 2 times the v5 half-width of 0.35)
ENV_ID = "SafetyPointSemicircle0-v6"
FPS = float(os.environ.get("SEMICIRCLE_TELEOP_FPS", "60"))
ACTION_SCALE = float(os.environ.get("SEMICIRCLE_TELEOP_SCALE", "0.8"))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(action, low, high).astype(np.float32)


def run_teleop(env_id: str, *, seed: int | None = 42) -> None:
    _log(
        f"env_show [wide track] {env_id!r}  MUJOCO_GL={os.environ.get('MUJOCO_GL', '')!r}  "
        f"FPS={FPS:g}  scale={ACTION_SCALE:g}"
    )
    _log("W/S front and rear A/D left and right R restart Q/ESC exit")
    _log("Create environment...")
    env = safety_gymnasium.make(env_id, render_mode="human")
    _log("reset() (may be slower the first time)...")
    reset_kw = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**reset_kw)

    pressed = {"w": False, "a": False, "s": False, "d": False}
    running = {"ok": True}
    ep = 1

    root = tk.Tk()
    root.title(f"Semicircle wide track ({env_id})")
    root.geometry("520x120")
    tk.Label(
        root,
        text="Wide Track v6 | W/S/A/D Drive R Restart Q/ESC Exit",
    ).pack(pady=20)
    root.focus_force()

    def on_press(event) -> None:
        key = (event.keysym or "").lower()
        if key in pressed:
            pressed[key] = True
        elif key == "r":
            nonlocal obs, ep
            obs, _ = env.reset(**reset_kw)
            ep += 1
            _log(f"Manual reset → episode {ep}")
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
                f"episode {ep} End: term={term} trunc={trunc} "
                f"reward={float(reward):.3f} cost={float(cost):.3f}"
            )
            obs, _ = env.reset(**reset_kw)
            ep += 1

        next_frame += frame_dt
        sleep_for = next_frame - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_frame = time.perf_counter()

    root.destroy()
    env.close()
    _log("Exited")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Semicircle wide track v6 manual preview")
    p.add_argument(
        "--robot",
        type=str,
        default="Point",
        choices=["Point", "Car", "Racecar"],
        help="Agent type (default Point)",
    )
    p.add_argument("--env-id", type=str, default=None, help="full env id, overrides --robot")
    p.add_argument("--seed", type=int, default=42, help="reset random seed; pass -1 to indicate unfixed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env_id = args.env_id or f"Safety{args.robot}Semicircle0-v6"
    seed = None if args.seed < 0 else args.seed
    run_teleop(env_id, seed=seed)


if __name__ == "__main__":
    main()
