"""Go2 PPO strategic rendering: following camera directly behind + overhead HUD.

On-policy full ``Full_8000.pt`` is loaded by default, with fixed instructions vx=0.5, vy=0, yaw=0.
Birth pose is consistent with training ``cfgs.pkl`` (front leg thigh 0.8 / hind leg 1.0 rad, see HUD).
The camera follows directly behind the horizontal plane; the gray box shows the birth angle, number of steps, gradual reward, cumulative return, and speed tracking.

Run::

    python unitree_go2/render.py
    python unitree_go2/render.py --cpu
    python unitree_go2/render.py --no-reward # Turn off the reward item (not comparable to data_show)"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import re
import sys
import tkinter as tk
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_LOCOMOTION = _ROOT.parent / "Genesis" / "examples" / "locomotion"
if _LOCOMOTION.is_dir():
    sys.path.insert(0, str(_LOCOMOTION))
else:
    raise FileNotFoundError(f"Genesis locomotion not found: {_LOCOMOTION}")

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install rsl-rl-lib>=5.0.0") from e
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import genesis as gs  # noqa: E402
import genesis.utils.geom as gu  # noqa: E402
from genesis.utils.geom import transform_by_quat  # noqa: E402
from go2_env import Go2Env  # noqa: E402

DEFAULT_MODEL_PT = (
    _ROOT
    / "DTRL-On/runs/onpolicy/DTRL-On_full_20260516_092741_449462/DTRL-On_full_8000.pt"
)

VX_CMD = 0.5
VY_CMD = 0.0
YAW_CMD = 0.0

HUD_BG = "#505050"
HUD_FG = "#f2f2f2"
HUD_ALPHA = 0.82
HUD_FONT_SIZE = 13
HUD_PAD_X = 20
HUD_PAD_Y = 16
HUD_BORDER = 2
DEFAULT_HUD_OFFSET_X = 200
DEFAULT_HUD_OFFSET_Y = 100
CAM_BACK = 2.5
CAM_SIDE = 0.0
CAM_HEIGHT = 1.0
LOOKAT_FORWARD = 0.15
LOOKAT_HEIGHT = 0.38
HEAD_FORWARD = 0.24
HEAD_SIDE = 0.12
HEAD_HEIGHT = 0.40
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_VCMD_NORM_EPS = 1e-8

# Training stance angle (FR,FL,RR,RL × hip,thigh,calf) consistent with cfgs.pkl/data_show
TRAIN_DEFAULT_DOF_POS = np.array(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=np.float64,
)
_LEG_LABELS = ("FR", "FL", "RR", "RL")


def _default_dof_pos_np(env: Go2Env) -> np.ndarray:
    """Go2Env.default_dof_pos is a (12,) shared vector, not (num_envs, 12)."""
    dof = env.default_dof_pos.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if dof.size != 12:
        raise ValueError(f"default_dof_pos should be 12 dimensions, current shape={dof.shape}")
    return dof


def _format_spawn_pose(dof: np.ndarray, joint_names: list[str] | None = None) -> str:
    """For HUD: highlight the asymmetry of the thigh angles of the front and rear legs."""
    lines = ["Spawn (cfgs.pkl train pose)"]
    for i, leg in enumerate(_LEG_LABELS):
        h, t, c = float(dof[i * 3]), float(dof[i * 3 + 1]), float(dof[i * 3 + 2])
        lines.append(f"  {leg}: hip={h:+.1f}  thigh={t:+.1f}  calf={c:+.1f}")
    if joint_names:
        lines.append(f"  order: {', '.join(joint_names[:3])} …")
    return "\n".join(lines)


def _spawn_pose_header(env: Go2Env) -> str:
    dof = _default_dof_pos_np(env)
    names = list(env.env_cfg.get("joint_names", []))
    header = _format_spawn_pose(dof, names if names else None)
    diff = float(np.max(np.abs(dof - TRAIN_DEFAULT_DOF_POS)))
    if diff > 1e-4:
        header += f"\n  !! vs TRAIN_DEFAULT diff max={diff:.4f}"
    else:
        header += "\n  ✓ matches TRAIN_DEFAULT_DOF_POS"
    return header


def _velocity_tracking_errors(v_cmd: np.ndarray, v_real: np.ndarray) -> tuple[float, float, float]:
    """Ex, Ey（m/s）；E% = 100 * ||v_real - v_cmd||^2 / ||v_cmd||^2。"""
    ex = float(v_real[0] - v_cmd[0])
    ey = float(v_real[1] - v_cmd[1])
    v_cmd_norm_sq = float(v_cmd[0] ** 2 + v_cmd[1] ** 2)
    if v_cmd_norm_sq < _VCMD_NORM_EPS:
        e_pct = float("nan")
    else:
        err_sq = ex * ex + ey * ey
        e_pct = 100.0 * err_sq / v_cmd_norm_sq
    return ex, ey, e_pct


def _pin_commands(env: Go2Env, fixed: torch.Tensor) -> None:
    def _pinned(envs_idx=None) -> None:
        if envs_idx is None:
            env.commands.copy_(fixed)
        else:
            idx = envs_idx.nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                env.commands[idx] = fixed[idx]

    env._resample_commands = _pinned  # type: ignore[method-assign]


def _resolve_run_dir_and_ckpt(model_path: Path) -> tuple[Path, int]:
    p = model_path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Weight not found: {p}")
    m = re.fullmatch(r"model_(\d+)\.pt", p.name)
    if not m:
        raise ValueError(f"The weight file name must be model_<iter>.pt: {p.name}")
    return p.parent, int(m.group(1))


def _yaw_basis_xy(env: Go2Env) -> tuple[np.ndarray, np.ndarray]:
    """The projection of the forward direction of the body on the horizontal plane (ignore roll/pitch and ensure the horizon is horizontal)."""
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device)
    fwd = transform_by_quat(ex.unsqueeze(0), env.base_quat[0:1])[0].detach().cpu().numpy()
    fwd_xy = fwd[:2].astype(np.float64)
    n = float(np.linalg.norm(fwd_xy))
    if n < 1e-6:
        fwd_xy = np.array([1.0, 0.0], dtype=np.float64)
    else:
        fwd_xy /= n
    right_xy = np.array([-fwd_xy[1], fwd_xy[0]], dtype=np.float64)
    return fwd_xy, right_xy


def _chase_camera_targets(env: Go2Env) -> tuple[np.ndarray, np.ndarray]:
    """Looking straight back; position/orientation are calculated on the horizontal plane."""
    base = env.base_pos[0].detach().cpu().numpy()
    fwd_xy, right_xy = _yaw_basis_xy(env)
    pos = np.array(
        [
            base[0] - CAM_BACK * fwd_xy[0] + CAM_SIDE * right_xy[0],
            base[1] - CAM_BACK * fwd_xy[1] + CAM_SIDE * right_xy[1],
            base[2] + CAM_HEIGHT,
        ],
        dtype=np.float64,
    )
    lookat = np.array(
        [
            base[0] + LOOKAT_FORWARD * fwd_xy[0],
            base[1] + LOOKAT_FORWARD * fwd_xy[1],
            base[2] + LOOKAT_HEIGHT,
        ],
        dtype=np.float64,
    )
    return pos, lookat


def _update_behind_camera(env: Go2Env) -> None:
    """Hard follow behind (no smoothing), the world up is fixed, and the picture is not distorted."""
    viewer = env.scene.visualizer.viewer
    viewer._followed_entity = None

    cam_pos, lookat = _chase_camera_targets(env)
    viewer._camera_up = WORLD_UP.copy()
    pose = gu.pos_lookat_up_to_T(cam_pos, lookat, WORLD_UP)
    viewer.set_camera_pose(pose=pose)


def _viewer_window_rect(prv) -> tuple[int, int, int, int] | None:
    try:
        if getattr(prv, "_window", None) is None:
            return None
        wx, wy = prv.get_location()
        ww, wh = prv.get_size()
        return int(wx), int(wy), int(ww), int(wh)
    except Exception:
        return None


def _project_world_to_screen(env: Go2Env, world_pt: np.ndarray) -> tuple[int, int] | None:
    gv = env.scene.visualizer.viewer
    prv = getattr(gv, "_pyrender_viewer", None)
    if prv is None:
        return None

    win = _viewer_window_rect(prv)
    if win is None:
        return None
    wx, wy, ww, wh = win

    mtx = np.asarray(prv._trackball._n_pose, dtype=np.float64)
    cam_pos = mtx[:3, 3]
    forward = -mtx[:3, 2]
    right = mtx[:3, 0]
    up = mtx[:3, 1]

    v = world_pt.astype(np.float64) - cam_pos
    depth = float(np.dot(v, forward))
    if depth < 0.08:
        return None

    x_ndc = float(np.dot(v, right)) / depth
    y_ndc = float(np.dot(v, up)) / depth

    tan_half = np.tan(0.5 * np.deg2rad(float(gv.camera_fov)))
    vw, vh = float(prv.viewport_size[0]), float(prv.viewport_size[1])
    sx = 0.5 * vw + x_ndc * vh / (2.0 * tan_half)
    sy = 0.5 * vh + y_ndc * vh / (2.0 * tan_half)

    scale_x = ww / max(vw, 1.0)
    scale_y = wh / max(vh, 1.0)
    screen_x = int(wx + sx * scale_x)
    screen_y = int(wy + wh - sy * scale_y)
    return screen_x, screen_y


def _robot_hud_anchor(env: Go2Env) -> np.ndarray:
    base = env.base_pos[0].detach().cpu().numpy()
    fwd_xy, right_xy = _yaw_basis_xy(env)
    return np.array(
        [
            base[0] + HEAD_FORWARD * fwd_xy[0] + HEAD_SIDE * right_xy[0],
            base[1] + HEAD_FORWARD * fwd_xy[1] + HEAD_SIDE * right_xy[1],
            base[2] + HEAD_HEIGHT,
        ],
        dtype=np.float64,
    )


class VelocityOverlay:
    """Translucent gray box: training birth posture + gradual reward/return + speed tracking."""

    def __init__(
        self,
        *,
        spawn_header: str,
        offset_x: int,
        offset_y: int,
        show_rewards: bool,
        debug: bool = False,
    ) -> None:
        self._spawn_header = spawn_header
        self._show_rewards = show_rewards
        self._offset_x = int(offset_x)
        self._offset_y = int(offset_y)
        self._debug = debug
        self._debug_counter = 0
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
        try:
            self._win.destroy()
        except tk.TclError:
            pass
        try:
            self._root.destroy()
        except tk.TclError:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def update(
        self,
        v_cmd: np.ndarray,
        v_real: np.ndarray,
        *,
        step: int,
        episode: int,
        reward: float,
        return_sum: float,
        action: np.ndarray | None = None,
    ) -> None:
        if self.closed:
            return
        ex, ey, e_pct = _velocity_tracking_errors(v_cmd, v_real)
        if np.isnan(e_pct):
            e_str = "E = n/a (|v_cmd|≈0)"
        else:
            e_str = f"E = {e_pct:.2f}%"
        parts = [self._spawn_header, ""]
        if self._show_rewards:
            act_hint = ""
            if action is not None:
                act_hint = f"  |a|={float(np.linalg.norm(action)):.3f}"
            parts.append(
                f"ep={episode}  step={step}  r={reward:+.4f}  ret={return_sum:.3f}{act_hint}"
            )
        else:
            parts.append(f"ep={episode}  step={step}  (reward off, use w/o --no-reward)")
        parts.extend(
            [
                f"V_cmd  = [{v_cmd[0]:+.3f}, {v_cmd[1]:+.3f}] m/s",
                f"V_real = [{v_real[0]:+.3f}, {v_real[1]:+.3f}] m/s",
                f"Ex={ex:+.3f}  Ey={ey:+.3f}  {e_str}",
            ]
        )
        self._label.config(text="\n".join(parts))

    def follow_screen_pos(self, screen_x: int, screen_y: int) -> None:
        if self.closed:
            return
        self._win.update_idletasks()
        w = max(self._win.winfo_reqwidth(), self._win.winfo_width(), 1)
        h = max(self._win.winfo_reqheight(), self._win.winfo_height(), 1)
        x = int(screen_x) + self._offset_x
        y = int(screen_y) + self._offset_y
        sw = self._win.winfo_screenwidth()
        sh = self._win.winfo_screenheight()
        x = max(0, min(x, sw - w))
        y = max(0, min(y, sh - h))
        self._win.geometry(f"{w}x{h}+{x}+{y}")

        if self._debug:
            self._debug_counter += 1
            if self._debug_counter % 60 == 1:
                print(
                    f"[HUD] anchor=({screen_x},{screen_y}) offset=({self._offset_x},{self._offset_y}) "
                    f"-> top_left=({x},{y}) size=({w}x{h})",
                    flush=True,
                )

    def pump(self) -> None:
        if self.closed:
            return
        self._root.update_idletasks()
        self._root.update()


def run_render(
    model_path: Path,
    *,
    use_cpu: bool,
    seed: int,
    hud_offset_x: int,
    hud_offset_y: int,
    hud_debug: bool,
    show_rewards: bool,
) -> None:
    log_dir, ckpt = _resolve_run_dir_and_ckpt(model_path)
    cfg_path = log_dir / "cfgs.pkl"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"not found {cfg_path}")

    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg = dict(reward_cfg)
    if not show_rewards:
        reward_cfg["reward_scales"] = {}

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
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), str(log_dir), device=gs.device)
    runner.load(os.path.join(str(log_dir), f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    fixed = torch.tensor([[VX_CMD, VY_CMD, YAW_CMD]], dtype=gs.tc_float, device=gs.device)
    _pin_commands(env, fixed)

    spawn_header = _spawn_pose_header(env)
    overlay = VelocityOverlay(
        spawn_header=spawn_header,
        offset_x=hud_offset_x,
        offset_y=hud_offset_y,
        show_rewards=show_rewards,
        debug=hud_debug,
    )
    last_screen: tuple[int, int] | None = None

    dof = _default_dof_pos_np(env)
    print(f"[render] model={model_path}")
    print(f"[render] ckpt={ckpt}  gs_seed={seed}  reward={'on (same as data_show)' if show_rewards else 'off'}")
    print(f"[render] command vx={VX_CMD}, vy={VY_CMD}, yaw={YAW_CMD}(Fixed, no random resampling)")
    print("[render] born default_dof_pos (rad):")
    for i, leg in enumerate(_LEG_LABELS):
        print(
            f"  {leg}: hip={dof[i*3]:+.3f}  thigh={dof[i*3+1]:+.3f}  calf={dof[i*3+2]:+.3f}"
        )
    print(f"[render] vector: {np.array2string(dof, precision=3, separator=', ')}")
    print(
        f"[render] HUD offset offset_x={hud_offset_x} offset_y={hud_offset_y} "
        f"(positive=right/bottom; after changing the file, you must restart the process, or use --hud-offset-x/y)"
    )
    print("[render] Follow directly behind. Close the gray box or Ctrl+C to exit.")

    obs_dict = env.reset()
    env.commands.copy_(fixed)
    step = 0
    episode = 1
    return_sum = 0.0
    last_reward = 0.0
    try:
        with torch.no_grad():
            while not overlay.closed:
                actions = policy(obs_dict)
                act_np = actions.detach().cpu().numpy().reshape(-1)
                obs_dict, rews, dones, _extras = env.step(actions)
                env.commands.copy_(fixed)
                last_reward = float(rews[0].item())
                return_sum += last_reward
                step += 1

                _update_behind_camera(env)

                v_cmd = env.commands[0, :2].detach().cpu().numpy()
                v_real = env.base_lin_vel[0, :2].detach().cpu().numpy()
                overlay.update(
                    v_cmd,
                    v_real,
                    step=step,
                    episode=episode,
                    reward=last_reward,
                    return_sum=return_sum,
                    action=act_np,
                )

                if bool(dones[0].item()):
                    print(
                        f"[render] ep={episode} End step={step}  return={return_sum:.4f}  "
                        f"(Can be compared with data_show when it is the same as cfg)",
                        flush=True,
                    )
                    obs_dict = env.reset()
                    env.commands.copy_(fixed)
                    step = 0
                    return_sum = 0.0
                    last_reward = 0.0
                    episode += 1

                anchor = _robot_hud_anchor(env)
                screen_xy = _project_world_to_screen(env, anchor)
                if screen_xy is not None:
                    last_screen = screen_xy
                if last_screen is not None:
                    overlay.follow_screen_pos(*last_screen)

                overlay.pump()
    except KeyboardInterrupt:
        print("\n[render] Interrupted")
    finally:
        try:
            gs.destroy()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Go2 PPO rendering (camera + label box following)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PT))
    p.add_argument("--hud-offset-x", type=int, default=DEFAULT_HUD_OFFSET_X)
    p.add_argument("--hud-offset-y", type=int, default=DEFAULT_HUD_OFFSET_Y)
    p.add_argument("--hud-debug", action="store_true")
    p.add_argument(
        "--no-reward",
        action="store_true",
        help="Clear reward_scales (ret in HUD is always 0, not comparable to data_show)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Weight not found: {model_path}")
    run_render(
        model_path,
        use_cpu=args.cpu,
        seed=args.seed,
        hud_offset_x=args.hud_offset_x,
        hud_offset_y=args.hud_offset_y,
        hud_debug=args.hud_debug,
        show_rewards=not args.no_reward,
    )


if __name__ == "__main__":
    main()
