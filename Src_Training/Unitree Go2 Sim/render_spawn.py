"""Go2 birth posture maintenance: no strategy, PD target is always default_dof_pos of cfgs.pkl (equivalent to action=0).

Motor combination is consistent with training (front leg thigh 0.8 / hind leg 1.0 rad). **Free perspective** in the Genesis window (mouse drag to rotate/translate),
The small box in the upper left corner displays the target angle, current joint angle and attitude error.

Run::

    python unitree_go2/render_spawn.py
    python unitree_go2/render_spawn.py --cpu"""

from __future__ import annotations

import argparse
import pickle
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_LOCOMOTION = _ROOT.parent / "Genesis" / "examples" / "locomotion"
if _LOCOMOTION.is_dir():
    sys.path.insert(0, str(_LOCOMOTION))
else:
    raise FileNotFoundError(f"Genesis locomotion not found: {_LOCOMOTION}")

import genesis as gs
from go2_env import Go2Env

DEFAULT_RUN_DIR = (
    _ROOT / "DTRL-On/runs/onpolicy/DTRL-On_full_20260516_092741_449462"
)

HUD_BG = "#505050"
HUD_FG = "#f2f2f2"
HUD_ALPHA = 0.82
HUD_FONT_SIZE = 12
HUD_PAD_X = 16
HUD_PAD_Y = 12
HUD_BORDER = 2
DEFAULT_HUD_MARGIN_X = 12
DEFAULT_HUD_MARGIN_Y = 12
_LEG_LABELS = ("FR", "FL", "RR", "RL")


def _default_dof_pos_np(env: Go2Env) -> np.ndarray:
    dof = env.default_dof_pos.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if dof.size != 12:
        raise ValueError(f"default_dof_pos should be 12 dimensions, current shape={dof.shape}")
    return dof


def _format_spawn_pose(dof: np.ndarray) -> str:
    lines = ["Target q* (default_dof_pos)"]
    for i, leg in enumerate(_LEG_LABELS):
        h, t, c = float(dof[i * 3]), float(dof[i * 3 + 1]), float(dof[i * 3 + 2])
        lines.append(f"  {leg}: hip={h:+.3f}  thigh={t:+.3f}  calf={c:+.3f}")
    return "\n".join(lines)


def _format_current_q(q: np.ndarray, joint_names: list[str] | None) -> str:
    lines = ["Current q (rad)"]
    if joint_names and len(joint_names) == 12:
        for name, val in zip(joint_names, q.reshape(-1), strict=True):
            short = name.replace("_joint", "")
            lines.append(f"  {short:16s} {float(val):+.4f}")
    else:
        for i, leg in enumerate(_LEG_LABELS):
            h, t, c = float(q[i * 3]), float(q[i * 3 + 1]), float(q[i * 3 + 2])
            lines.append(f"  {leg}: hip={h:+.4f}  thigh={t:+.4f}  calf={c:+.4f}")
    return "\n".join(lines)


def _noop_resample(env: Go2Env) -> None:
    env._resample_commands = lambda envs_idx=None: None  # type: ignore[method-assign]


def _enable_free_camera(env: Go2Env) -> None:
    """Does not follow the body and retains the Genesis default trackball free perspective."""
    viewer = env.scene.visualizer.viewer
    viewer._followed_entity = None


def _viewer_window_rect(env: Go2Env) -> tuple[int, int, int, int] | None:
    prv = getattr(env.scene.visualizer.viewer, "_pyrender_viewer", None)
    if prv is None:
        return None
    return _pyrender_window_rect(prv)


def _pyrender_window_rect(prv) -> tuple[int, int, int, int] | None:
    try:
        if getattr(prv, "_window", None) is None:
            return None
        wx, wy = prv.get_location()
        ww, wh = prv.get_size()
        return int(wx), int(wy), int(ww), int(wh)
    except Exception:
        return None


class SpawnOverlay:
    def __init__(self, *, header: str, margin_x: int, margin_y: int) -> None:
        self._header = header
        self._margin_x = int(margin_x)
        self._margin_y = int(margin_y)
        self._root = tk.Tk()
        self._root.withdraw()
        self._win = tk.Toplevel(self._root)
        self._win.overrideredirect(True)
        self._win.attributes("-topmost", True)
        try:
            self._win.attributes("-alpha", HUD_ALPHA)
        except tk.TclError:
            pass
        frame = tk.Frame(
            self._win,
            bg=HUD_BG,
            padx=HUD_PAD_X,
            pady=HUD_PAD_Y,
            highlightthickness=HUD_BORDER,
            highlightbackground="#888888",
        )
        frame.pack()
        self._label = tk.Label(
            frame,
            justify=tk.LEFT,
            font=("DejaVu Sans Mono", HUD_FONT_SIZE),
            fg=HUD_FG,
            bg=HUD_BG,
            anchor="w",
        )
        self._label.pack()
        self._closed = False
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self) -> None:
        self._closed = True
        for w in (self._win, self._root):
            try:
                w.destroy()
            except tk.TclError:
                pass

    @property
    def closed(self) -> bool:
        return self._closed

    def update(
        self,
        *,
        step: int,
        kp: float,
        kd: float,
        dof_err_max: float,
        base_z: float,
        pitch: float,
        roll: float,
        current_q_text: str,
    ) -> None:
        if self.closed:
            return
        text = (
            f"{self._header}\n"
            f"{current_q_text}\n"
            f"PD kp={kp:g}  kd={kd:g}\n"
            f"step={step}  max|q-q*|={dof_err_max:.4f} rad\n"
            f"base_z={base_z:.3f} m  pitch={pitch:+.1f}°  roll={roll:+.1f}°\n"
            f"(Free perspective: mouse drag rotation/wheel zoom)"
        )
        self._label.config(text=text)

    def place_on_viewer(self, env: Go2Env) -> None:
        """Pasted in the upper left corner of the Genesis window, does not follow the robot."""
        if self.closed:
            return
        self._win.update_idletasks()
        w = max(self._win.winfo_reqwidth(), self._win.winfo_width(), 1)
        h = max(self._win.winfo_reqheight(), self._win.winfo_height(), 1)
        win = _viewer_window_rect(env)
        if win is not None:
            wx, wy, _, _ = win
            x = int(wx) + self._margin_x
            y = int(wy) + self._margin_y
        else:
            x, y = self._margin_x, self._margin_y
        sw = self._win.winfo_screenwidth()
        sh = self._win.winfo_screenheight()
        x = max(0, min(x, sw - w))
        y = max(0, min(y, sh - h))
        self._win.geometry(f"{w}x{h}+{x}+{y}")

    def pump(self) -> None:
        if self.closed:
            return
        self._root.update_idletasks()
        self._root.update()


def run_spawn(
    run_dir: Path,
    *,
    use_cpu: bool,
    seed: int,
    hud_margin_x: int,
    hud_margin_y: int,
    no_auto_reset: bool,
) -> None:
    cfg_path = run_dir / "cfgs.pkl"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"not found {cfg_path}")

    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, _train_cfg = pickle.load(f)

    env_cfg = dict(env_cfg)
    obs_cfg = dict(obs_cfg)
    reward_cfg = dict(reward_cfg)
    reward_cfg["reward_scales"] = {}
    command_cfg = dict(command_cfg)
    if no_auto_reset:
        env_cfg["episode_length_s"] = 1e9
        env_cfg["termination_if_pitch_greater_than"] = 1e9
        env_cfg["termination_if_roll_greater_than"] = 1e9

    backend = gs.cpu if use_cpu else gs.gpu
    gs.init(
        backend=backend,
        precision="32",
        logging_level="warning",
        seed=seed,
        performance_mode=True,
    )

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    _noop_resample(env)
    _enable_free_camera(env)

    target = _default_dof_pos_np(env)
    header = _format_spawn_pose(target)
    joint_names = list(env_cfg.get("joint_names", []))
    overlay = SpawnOverlay(header=header, margin_x=hud_margin_x, margin_y=hud_margin_y)

    kp = float(env_cfg.get("kp", 20))
    kd = float(env_cfg.get("kd", 0.5))
    zero_action = torch.zeros((1, env.num_actions), dtype=gs.tc_float, device=gs.device)

    print(f"[render_spawn] run_dir={run_dir}")
    print(f"[render_spawn] No strategy; action=0 per step → target angle = default_dof_pos")
    print(f"[render_spawn] Free perspective (mouse operation Genesis window)")
    print(f"[render_spawn] PD kp={kp} kd={kd}  gs_seed={seed}")
    print("[render_spawn] default_dof_pos (rad):")
    for i, leg in enumerate(_LEG_LABELS):
        print(
            f"  {leg}: hip={target[i*3]:+.3f}  thigh={target[i*3+1]:+.3f}  calf={target[i*3+2]:+.3f}"
        )
    print(f"[render_spawn] vector: {np.array2string(target, precision=3, separator=', ')}")
    if no_auto_reset:
        print("[render_spawn] Turned off dump/timeout automatic reset (--allow-reset can restore)")
    print("[render_spawn] Close gray box or Ctrl+C to exit.")

    env.reset()
    step = 0
    try:
        while not overlay.closed:
            _ = env.step(zero_action)
            step += 1

            q = env.dof_pos[0].detach().cpu().numpy().astype(np.float64)
            err = np.abs(q - target)
            base_z = float(env.base_pos[0, 2].item())
            pitch = float(env.base_euler[0, 1].item())
            roll = float(env.base_euler[0, 0].item())

            overlay.update(
                step=step,
                kp=kp,
                kd=kd,
                dof_err_max=float(err.max()),
                base_z=base_z,
                pitch=pitch,
                roll=roll,
                current_q_text=_format_current_q(q, joint_names if joint_names else None),
            )
            overlay.place_on_viewer(env)
            overlay.pump()
    except KeyboardInterrupt:
        print("\n[render_spawn] has been interrupted")
    finally:
        try:
            gs.destroy()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Go2 birth posture PD maintain (no strategy)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--run-dir",
        type=str,
        default=str(DEFAULT_RUN_DIR),
        help="Training run directory containing cfgs.pkl",
    )
    p.add_argument("--hud-margin-x", type=int, default=DEFAULT_HUD_MARGIN_X, help="The horizontal margin of the HUD relative to the upper left corner of the Genesis window")
    p.add_argument("--hud-margin-y", type=int, default=DEFAULT_HUD_MARGIN_Y, help="The vertical margin of the HUD relative to the upper left corner of the Genesis window")
    p.add_argument(
        "--allow-reset",
        action="store_true",
        help="Allow env to automatically reset after dumping/timeout (off by default, easy to observe whether it stands still)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_spawn(
        run_dir,
        use_cpu=args.cpu,
        seed=args.seed,
        hud_margin_x=args.hud_margin_x,
        hud_margin_y=args.hud_margin_y,
        no_auto_reset=not args.allow_reset,
    )


if __name__ == "__main__":
    main()
