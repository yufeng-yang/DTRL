"""Go2 FlashSAC EENN three-port random switching rendering.

Same as datashow.py: loads FlashSACEENNActor from union checkpoint,
Switch evenly and randomly between e1/e2/efull every N steps, running for a total of 1000 steps.
Direct rear following camera + speed HUD. Task: go2_walk_easy, vx=0.5 m/s go straight.

Run::

    python unitree_go2/evaluation/render.py
    python unitree_go2/evaluation/render.py --cpu"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tkinter as tk
import types
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.geom import transform_by_quat

_EVAL = Path(__file__).resolve().parent
_ROOT = _EVAL.parent
_FLASHSAC = _ROOT.parents[1] / "extra_resources" / "FlashSAC"

DEFAULT_CHECKPOINT = (
    _ROOT
    / "DTRL-Off/runs/DTRL-Off_20260517_091613/checkpoints"
    / "seed42-0517-091635/step48825"
)

VX_CMD = 0.5
VY_CMD = 0.0
YAW_CMD = 0.0

ExitName = Literal["e1", "e2", "efull"]
EXIT_TAGS: tuple[ExitName, ...] = ("e1", "e2", "efull")

GS_SEED = 0
N_STEPS = 1000
SWITCH_INTERVAL = 10

HUD_BG = "#505050"
HUD_FG = "#f2f2f2"
HUD_ALPHA = 0.82
HUD_FONT_SIZE = 15
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


def _genesis_teardown() -> None:
    try:
        gs.destroy()
    except Exception:
        pass


def _velocity_tracking_errors(v_cmd: np.ndarray, v_real: np.ndarray) -> tuple[float, float, float]:
    ex = float(v_real[0] - v_cmd[0])
    ey = float(v_real[1] - v_cmd[1])
    v_cmd_norm_sq = float(v_cmd[0] ** 2 + v_cmd[1] ** 2)
    if v_cmd_norm_sq < _VCMD_NORM_EPS:
        e_pct = float("nan")
    else:
        e_pct = 100.0 * (ex * ex + ey * ey) / v_cmd_norm_sq
    return ex, ey, e_pct


def _yaw_basis_xy(env: Any) -> tuple[np.ndarray, np.ndarray]:
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


def _chase_camera_targets(env: Any) -> tuple[np.ndarray, np.ndarray]:
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


def _update_behind_camera(env: Any) -> None:
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


def _project_world_to_screen(env: Any, world_pt: np.ndarray) -> tuple[int, int] | None:
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
    return int(wx + sx * scale_x), int(wy + wh - sy * scale_y)


def _robot_hud_anchor(env: Any) -> np.ndarray:
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
    def __init__(self, *, offset_x: int, offset_y: int) -> None:
        self._offset_x = int(offset_x)
        self._offset_y = int(offset_y)
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
        exit_tag: str,
        step: int,
    ) -> None:
        if self.closed:
            return
        ex, ey, e_pct = _velocity_tracking_errors(v_cmd, v_real)
        e_str = "E = n/a" if np.isnan(e_pct) else f"E = {e_pct:.2f}%"
        text = (
            f"exit = {exit_tag}  step = {step}\n"
            f"V_cmd  = [{v_cmd[0]:+.3f}, {v_cmd[1]:+.3f}] m/s\n"
            f"V_real = [{v_real[0]:+.3f}, {v_real[1]:+.3f}] m/s\n"
            f"Ex={ex:+.3f}  Ey={ey:+.3f}  {e_str}"
        )
        self._label.config(text=text)

    def follow_screen_pos(self, screen_x: int, screen_y: int) -> None:
        if self.closed:
            return
        self._win.update_idletasks()
        w = max(self._win.winfo_reqwidth(), self._win.winfo_width(), 1)
        h = max(self._win.winfo_reqheight(), self._win.winfo_height(), 1)
        x = max(0, min(int(screen_x) + self._offset_x, self._win.winfo_screenwidth() - w))
        y = max(0, min(int(screen_y) + self._offset_y, self._win.winfo_screenheight() - h))
        self._win.geometry(f"{w}x{h}+{x}+{y}")

    def pump(self) -> None:
        if self.closed:
            return
        self._root.update_idletasks()
        self._root.update()


def _resolve_actor_pt(path: Path) -> Path:
    p = path.expanduser().resolve()
    if p.is_file() and p.suffix == ".pt":
        return p
    direct = p / "actor.pt"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"actor.pt not found: {p}")


def _import_flashsac_eenn_actor_class():
    flashsac = _FLASHSAC.resolve()
    if str(flashsac) not in sys.path:
        sys.path.insert(0, str(flashsac))

    def _load_py_module(dotted: str, file_path: Path):
        if dotted in sys.modules:
            return sys.modules[dotted]
        spec = importlib.util.spec_from_file_location(dotted, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module: {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    for pkg in (
        "flash_rl",
        "flash_rl.agents",
        "flash_rl.agents.flashSAC",
        "flash_rl.agents.utils",
    ):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    _load_py_module(
        "flash_rl.agents.utils.distribution",
        flashsac / "flash_rl/agents/utils/distribution.py",
    )
    _load_py_module(
        "flash_rl.agents.flashSAC.layer",
        flashsac / "flash_rl/agents/flashSAC/layer.py",
    )
    eenn_mod = _load_py_module(
        "flash_rl.agents.flashSAC.eenn_actor",
        flashsac / "flash_rl/agents/flashSAC/eenn_actor.py",
    )
    return eenn_mod.FlashSACEENNActor


def _load_actor(actor_pt: Path, device: torch.device):
    FlashSACEENNActor = _import_flashsac_eenn_actor_class()
    ckpt = torch.load(actor_pt, map_location=device, weights_only=False)
    raw = ckpt.get("network_state_dict", ckpt.get("actor", ckpt))
    if not isinstance(raw, dict):
        raise TypeError(f"Unable to parse checkpoint: {actor_pt}")
    state = {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v for k, v in raw.items()
    }
    obs_dim, act_dim = 45, 12
    for key, tensor in state.items():
        if key == "backbone.0.weight":
            obs_dim = int(tensor.shape[1])
        if key == "exit1.mean_w.w.weight":
            act_dim = int(tensor.shape[0])
    actor = FlashSACEENNActor(obs_dim, act_dim).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


@torch.no_grad()
def _exit_action(actor, obs: np.ndarray, exit_name: ExitName, device: torch.device) -> np.ndarray:
    """Exactly the same as datashow.py."""
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if exit_name == "e1":
        h0, _, _ = actor._features(x)
        mean, _ = actor.exit1.get_mean_and_std(h0, training=False)
    elif exit_name == "e2":
        _, h2, _ = actor._features(x)
        mean, _ = actor.exit2.get_mean_and_std(h2, training=False)
    else:
        mean, _ = actor.get_mean_and_std(x, training=False)
    return torch.tanh(mean).cpu().numpy()


def _import_go2_walk_easy_mod():
    flashsac = _FLASHSAC.resolve()
    if str(flashsac) not in sys.path:
        sys.path.insert(0, str(flashsac))
    for pkg in ("flash_rl", "flash_rl.envs", "flash_rl.envs.genesis_envs"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
    path = flashsac / "flash_rl/envs/genesis_envs/go2_walk_easy.py"
    spec = importlib.util.spec_from_file_location("flash_rl.envs.genesis_envs.go2_walk_easy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flash_rl.envs.genesis_envs.go2_walk_easy"] = mod
    spec.loader.exec_module(mod)
    return mod


def _create_env(*, show_viewer: bool):
    mod = _import_go2_walk_easy_mod()
    env_cfg, obs_cfg, reward_cfg, command_cfg = mod.get_cfgs(
        cmd_vx=VX_CMD, cmd_vy=VY_CMD, cmd_yaw=YAW_CMD
    )
    return mod.Go2WalkEasyEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )


def _pin_commands(env, fixed: torch.Tensor) -> None:
    def _pinned(envs_idx=None) -> None:
        if envs_idx is None:
            env.commands.copy_(fixed)
        else:
            idx = envs_idx.nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                env.commands[idx] = fixed[idx]

    env._resample_commands = _pinned  # type: ignore[method-assign]


def render_episode(
    *,
    checkpoint: Path,
    use_cpu: bool,
    device: torch.device,
    rng: np.random.Generator,
    switch_interval: int,
    hud_offset_x: int,
    hud_offset_y: int,
) -> None:
    actor_pt = _resolve_actor_pt(checkpoint)
    actor = _load_actor(actor_pt, device)

    _genesis_teardown()
    backend = gs.cpu if use_cpu else gs.gpu
    gs.init(
        backend=backend,
        precision="32",
        logging_level="warning",
        seed=GS_SEED,
        performance_mode=True,
    )

    env = _create_env(show_viewer=True)
    action_range = float(env.env_cfg["action_range"])
    fixed = torch.tensor([[VX_CMD, VY_CMD, YAW_CMD]], dtype=gs.tc_float, device=gs.device)
    _pin_commands(env, fixed)

    overlay = VelocityOverlay(offset_x=hud_offset_x, offset_y=hud_offset_y)
    last_screen: tuple[int, int] | None = None

    print(f"Weight: {actor_pt}")
    print(
        f"Task: vx={VX_CMD} | every {switch_interval} step randomly switches e1/e2/efull | "
        f"common {N_STEPS} step | Close HUD or Ctrl+C to exit"
    )

    obs_buf, _ = env.reset()
    env.commands.copy_(fixed)
    obs = obs_buf.detach().cpu().numpy()

    ep_ret = 0.0
    step = 0
    n_resets = 0
    exit_counts: dict[str, int] = {tag: 0 for tag in EXIT_TAGS}
    active: ExitName = EXIT_TAGS[int(rng.integers(0, len(EXIT_TAGS)))]

    try:
        while step < N_STEPS and not overlay.closed:
            if step % switch_interval == 0:
                active = EXIT_TAGS[int(rng.integers(0, len(EXIT_TAGS)))]

            act = _exit_action(actor, obs, active, device) * action_range
            exit_counts[active] += 1
            act_t = torch.as_tensor(act, dtype=gs.tc_float, device=gs.device)
            if act_t.ndim == 1:
                act_t = act_t.unsqueeze(0)
            obs_t, rews, dones, _extras = env.step(act_t)
            env.commands.copy_(fixed)
            obs = obs_t.detach().cpu().numpy()
            ep_ret += float(rews[0].item())
            step += 1

            if bool(dones[0].item()):
                n_resets += 1
                obs_buf, _ = env.reset()
                env.commands.copy_(fixed)
                obs = obs_buf.detach().cpu().numpy()

            _update_behind_camera(env)
            v_cmd = env.commands[0, :2].detach().cpu().numpy()
            v_real = env.base_lin_vel[0, :2].detach().cpu().numpy()
            overlay.update(v_cmd, v_real, exit_tag=active, step=step)
            screen_xy = _project_world_to_screen(env, _robot_hud_anchor(env))
            if screen_xy is not None:
                last_screen = screen_xy
            if last_screen is not None:
                overlay.follow_screen_pos(*last_screen)
            overlay.pump()
    except KeyboardInterrupt:
        print("\n[render] Interrupted")
    finally:
        _genesis_teardown()

    print(
        f"done | return={ep_ret:.4f} steps={step}/{N_STEPS} resets={n_resets} "
        f"exits={exit_counts}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 three-port random switching rendering (joint actor.pt)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="step directory or actor.pt (same as datashow.py)",
    )
    parser.add_argument("--rng-seed", type=int, default=42, help="Export switch RNG seed")
    parser.add_argument(
        "--switch-interval",
        type=int,
        default=SWITCH_INTERVAL,
        help=f"Randomly switch exits every few steps (default {SWITCH_INTERVAL}）",
    )
    parser.add_argument("--hud-offset-x", type=int, default=DEFAULT_HUD_OFFSET_X)
    parser.add_argument("--hud-offset-y", type=int, default=DEFAULT_HUD_OFFSET_Y)
    args = parser.parse_args()

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.exists() and not (ckpt / "actor.pt").exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    use_cpu = args.cpu
    device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(int(args.rng_seed))
    switch_interval = max(1, int(args.switch_interval))

    try:
        render_episode(
            checkpoint=ckpt,
            use_cpu=use_cpu,
            device=device,
            rng=rng,
            switch_interval=switch_interval,
            hud_offset_x=args.hud_offset_x,
            hud_offset_y=args.hud_offset_y,
        )
    finally:
        _genesis_teardown()


if __name__ == "__main__":
    main()
