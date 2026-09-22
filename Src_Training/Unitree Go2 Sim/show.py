"""Open the Go2 emulation in Genesis and open Viewer.

- Not added --stand-only: load checkpoint from the training directory; the default directory is shown below ``DEFAULT_LOG_DIR`` (please change it to your absolute path), which can also be overridden by ``--log-dir``.
- Added --stand-only: no network loading, zero-action standing demonstration.

  python show.py
  python show.py --log-dir /abs/path/to/runs/onpolicy/full_xxx
  python show.py --ckpt 100 --cpu
  python show.py --stand-only"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from importlib import metadata
from pathlib import Path

import torch

_LOCOMOTION = Path(__file__).resolve().parent.parent / "Genesis" / "examples" / "locomotion"
if _LOCOMOTION.is_dir():
    sys.path.insert(0, str(_LOCOMOTION))
else:
    raise FileNotFoundError(
        f"Genesis locomotion not found: {_LOCOMOTION}(Please run from within the RTSS repository or adjust the path)."
    )

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install rsl-rl-lib>=5.0.0 (required to load training weights).") from e
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import genesis as gs  # noqa: E402
from go2_env import Go2Env  # noqa: E402

# The run root directory of a certain full_train (containing cfgs.pkl and model_*.pt). Do not enter the model_xxx.pt file name here.
# If the weight file path is mistakenly written, the following logic will automatically use its parent directory and identify ckpt from the file name.
DEFAULT_LOG_DIR = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Unitree Go2 Sim/onpolicy/runs/onpolicy/full_20260424_103256_091127/model_149.pt"
)


def _resolve_run_dir_and_ckpt(log_in: Path, ckpt: int) -> tuple[Path, int]:
    """--log-dir can be the run root directory, or the absolute file path of a model_<iteration>.pt in this directory (iteration number is automatically recognized)."""
    p = log_in.expanduser().resolve()
    if p.is_file():
        m = re.fullmatch(r"model_(\d+)\.pt", p.name)
        if m:
            return p.parent, int(m.group(1))
        raise SystemExit(
            f"--log-dir points to non-checkpoint files: {p}; Please change to the run root directory, or the full path of a model_<number>.pt"
        )
    return p, ckpt


def get_cfgs():
    """Use --stand-only when there is no checkpoint (it is recommended that the command be all 0 when consistent with the go2_env training posture)."""
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg: dict = {"reward_scales": {}}
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Using the CPU backend")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="The absolute path to the training output directory, including cfgs.pkl and model_<ckpt>.pt (see DEFAULT_LOG_DIR in the script)",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=100,
        help="Load model_<ckpt>.pt (effective when --log-dir is a directory; if --log-dir is a .pt file path, it will be automatically parsed from the file name and this item will be ignored)",
    )
    parser.add_argument(
        "--stand-only",
        action="store_true",
        help="No strategy loaded, just zero actions (no inference done)",
    )
    args = parser.parse_args()

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning", performance_mode=True)

    if args.stand_only:
        env_cfg, obs_cfg, reward_cfg, command_cfg, _ = get_cfgs()
        train_cfg = None
    else:
        log_path, ckpt = _resolve_run_dir_and_ckpt(args.log_dir, args.ckpt)
        cfg_path = log_path / "cfgs.pkl"
        ckpt_path = log_path / f"model_{ckpt}.pt"
        if not cfg_path.is_file():
            raise FileNotFoundError(
                f"not found {cfg_path}. Please let --log-dir point to the run root directory (including cfgs.pkl),"
                f"Do not point to the model_*.pt files themselves, unless using the automatically recognized weight path logic below."
            )
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"not found {ckpt_path}. Please change --ckpt or confirm that the corresponding checkpoint has been saved under the run."
            )

        with open(cfg_path, "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
        reward_cfg["reward_scales"] = {}
        log_dir = str(log_path)

    if args.stand_only:
        env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
        actions = torch.zeros((env.num_envs, env.num_actions), device=gs.device, dtype=gs.tc_float)
        env.reset()
        with torch.no_grad():
            while True:
                _obs, _r, _d, _i = env.step(actions)
    else:
        assert train_cfg is not None
        env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=True,
        )
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
        runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
        policy = runner.get_inference_policy(device=gs.device)
        obs_dict = env.reset()
        with torch.no_grad():
            while True:
                actions = policy(obs_dict)
                obs_dict, _rews, _dones, _infos = env.step(actions)


if __name__ == "__main__":
    main()
