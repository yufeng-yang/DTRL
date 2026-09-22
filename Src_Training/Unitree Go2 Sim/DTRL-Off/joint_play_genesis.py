#!/usr/bin/env python3
"""FlashSAC EENN joint weight: Genesis **real-time window** rendering (**e1 port**, vx=0.5 m/s straight, no mp4 saved).

Usage::

    python unitree_go2/DTRL-Off/joint_play_genesis.py \\
        --checkpoint_path .../step29295"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

_SCRIPT = Path(__file__).resolve()
_OFFPOLICY = _SCRIPT.parent
_RTSS = _OFFPOLICY.parent.parent
_FLASHSAC = _RTSS / "extra_resources" / "FlashSAC"

VX_CMD = 0.5
VY_CMD = 0.0
YAW_CMD = 0.0
SEED = 42

CAM_BACK = 2.5
CAM_SIDE = 0.0
CAM_HEIGHT = 1.0
LOOKAT_FORWARD = 0.15
LOOKAT_HEIGHT = 0.38
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _ensure_uv_runtime(script_path: Path, argv: list[str]) -> None:
    flashsac_root = _FLASHSAC.resolve()
    if str(flashsac_root) not in sys.path:
        sys.path.insert(0, str(flashsac_root))
    try:
        import hydra  # noqa: F401
        import flash_rl  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    if os.environ.get("JOINT_PLAY_BOOTSTRAPPED") == "1":
        raise RuntimeError("The hydra/FlashSAC dependency is missing and uv boot fails.")

    uv_bin = str(Path.home() / ".local" / "bin" / "uv")
    env = os.environ.copy()
    env["JOINT_PLAY_BOOTSTRAPPED"] = "1"
    env["GO2_WALK_EASY_CMD_VX"] = str(VX_CMD)
    env["GO2_WALK_EASY_CMD_VY"] = str(VY_CMD)
    env["GO2_WALK_EASY_CMD_YAW"] = str(YAW_CMD)
    subprocess.run(
        [
            uv_bin,
            "run",
            "--project",
            str(flashsac_root),
            "--extra",
            "genesis",
            "python",
            str(script_path),
            *argv,
        ],
        check=True,
        env=env,
    )
    raise SystemExit(0)


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


def _create_env():
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
        show_viewer=True,
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


def _yaw_basis_xy(env) -> tuple[np.ndarray, np.ndarray]:
    import genesis as gs
    from genesis.utils.geom import transform_by_quat

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


def _chase_camera_targets(env) -> tuple[np.ndarray, np.ndarray]:
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


def _update_camera(env) -> None:
    import genesis.utils.geom as gu

    viewer = env.scene.visualizer.viewer
    viewer._followed_entity = None
    cam_pos, lookat = _chase_camera_targets(env)
    viewer._camera_up = WORLD_UP.copy()
    viewer.set_camera_pose(pose=gu.pos_lookat_up_to_T(cam_pos, lookat, WORLD_UP))


def _genesis_teardown() -> None:
    try:
        import genesis as gs

        gs.destroy()
    except Exception:
        pass


@torch.no_grad()
def _e1_actions(actor_net: Any, obs: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    h0, _, _ = actor_net._features(x)
    mean, _ = actor_net.exit1.get_mean_and_std(h0, training=False)
    return torch.tanh(mean).cpu().numpy()


def main() -> None:
    _ensure_uv_runtime(_SCRIPT, sys.argv[1:])
    import hydra
    from omegaconf import OmegaConf

    from flash_rl.agents import create_agent

    parser = argparse.ArgumentParser(description="EENN joint real-time rendering")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not (checkpoint_path / "actor.pt").is_file():
        raise SystemExit(f"actor.pt not found: {checkpoint_path}")

    os.environ["GO2_WALK_EASY_CMD_VX"] = str(VX_CMD)
    os.environ["GO2_WALK_EASY_CMD_VY"] = str(VY_CMD)
    os.environ["GO2_WALK_EASY_CMD_YAW"] = str(YAW_CMD)

    flashsac_root = _FLASHSAC.resolve()
    if str(flashsac_root) not in sys.path:
        sys.path.insert(0, str(flashsac_root))

    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    with hydra.initialize_config_dir(version_base=None, config_dir=str(flashsac_root / "configs")):
        cfg = hydra.compose(
            config_name="flashSAC_base",
            overrides=[
                f"seed={SEED}",
                "env=genesis",
                "env.env_name=go2-walk_easy",
                "agent=flashSAC_eenn",
                "agent.asymmetric_observation=true",
            ],
        )
    OmegaConf.resolve(cfg)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    obs_dim, act_dim = 45, 12
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
    )
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
    env_info = {"actor_observation_size": [obs_dim]}

    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(str(checkpoint_path))
    actor_net = agent._actor.network
    actor_net.eval()
    device = agent._device

    import genesis as gs

    _genesis_teardown()
    gs.init(
        backend=gs.gpu if torch.cuda.is_available() else gs.cpu,
        precision="32",
        logging_level="warning",
        seed=SEED,
        performance_mode=True,
    )

    env = _create_env()
    action_range = float(env.env_cfg["action_range"])
    fixed = torch.tensor([[VX_CMD, VY_CMD, YAW_CMD]], dtype=gs.tc_float, device=gs.device)
    _pin_commands(env, fixed)
    env.commands.copy_(fixed)

    obs_buf, _ = env.reset()
    obs = obs_buf.detach().cpu().numpy()
    reward_sum = 0.0
    max_steps = int(env.max_episode_length) + 50

    print(f"[joint_play] real-time rendering ckpt={checkpoint_path}")
    print(f"[joint_play] exit=e1  vx={VX_CMD:+.1f} m/s Close Genesis window ends")
    _update_camera(env)

    try:
        for step_i in range(max_steps):
            act = _e1_actions(actor_net, obs, device) * action_range
            act_t = torch.as_tensor(act, dtype=gs.tc_float, device=gs.device)
            obs_t, rews, dones, _ = env.step(act_t)
            env.commands.copy_(fixed)
            obs = obs_t.detach().cpu().numpy()
            reward_sum += float(rews[0].item())
            _update_camera(env)

            if bool(dones[0].item()):
                print(f"[joint_play] end return={reward_sum:.3f}  steps={step_i + 1}")
                return
        print(f"[joint_play] over {max_steps} step not finished")
    except KeyboardInterrupt:
        print("\n[joint_play] has been interrupted")
    finally:
        _genesis_teardown()


if __name__ == "__main__":
    main()
