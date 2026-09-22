"""Go2 PPO MMS-2: obs→512→256→128→out, trained from scratch (no pre-training loaded).

The default is 5000 PPO update iterations; checkpoint is saved and log is printed every 500 iter.

Run::

    cd ~/codespace/RTSS
    python unitree_go2/MMS/train_mms_2.py"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger

_SCRIPT_DIR = Path(__file__).resolve().parent
_RUNS_ROOT = _SCRIPT_DIR / "runs"
_GENESIS_LOCOMOTION_REL = Path("Genesis") / "examples" / "locomotion"


def _resolve_locomotion_dir(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_loc = os.environ.get("GENESIS_LOCOMOTION")
    if env_loc:
        candidates.append(Path(env_loc).expanduser())
    rtss_root = os.environ.get("RTSS_ROOT")
    if rtss_root:
        candidates.append(Path(rtss_root).expanduser() / _GENESIS_LOCOMOTION_REL)

    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents, Path.cwd(), *Path.cwd().parents):
        candidates.append(base / _GENESIS_LOCOMOTION_REL)

    tried: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        p = raw.resolve()
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        tried.append(key)
        if p.is_dir():
            return p

    raise FileNotFoundError(
        "Genesis locomotion not found. Tried:\n  "
        + "\n  ".join(tried[:12])
        + ("\n  ..." if len(tried) > 12 else "")
        + "\nCan set GENESIS_LOCOMOTION / RTSS_ROOT, or --genesis-locomotion."
    )


def _setup_go2_import(locomotion_dir: Path) -> None:
    loc = str(locomotion_dir.resolve())
    if loc not in sys.path:
        sys.path.insert(0, loc)


def _patch_logger_console_interval(logger: Logger, interval: int) -> None:
    if interval <= 1:
        return
    orig_log = logger.log

    def log(
        self,
        it: int,
        start_it: int,
        total_it: int,
        collect_time: float,
        learn_time: float,
        loss_dict: dict,
        learning_rate: float,
        action_std,
        rnd_weight=None,
        **kwargs,
    ) -> None:
        if it % interval == 0 or it == total_it - 1:
            orig_log(
                it,
                start_it,
                total_it,
                collect_time,
                learn_time,
                loss_dict,
                learning_rate,
                action_std,
                rnd_weight,
                **kwargs,
            )
            return
        if self.writer is not None:
            collection_size = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
            self.tot_timesteps += collection_size
            self.tot_time += collect_time + learn_time
        self.ep_extras.clear()

    logger.log = log.__get__(logger, Logger)  # type: ignore[method-assign]


def get_train_cfg(exp_name: str) -> dict:
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 500,
        "log_interval": 500,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs():
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
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [0.0, 0.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Go2 MMS-2 PPO: obs-512-256-128-out, trained from scratch")
    p.add_argument("-e", "--exp_name", type=str, default=None)
    p.add_argument("-B", "--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=5000, help="Number of PPO update iterations (default 5000)")
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--genesis-locomotion",
        type=str,
        default=None,
        help="Genesis/examples/locomotion directory; omitting it will automatically search upwards",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    locomotion_dir = _resolve_locomotion_dir(args.genesis_locomotion)
    _setup_go2_import(locomotion_dir)
    print(f"[MMS-2] Genesis locomotion: {locomotion_dir}")

    import genesis as gs  # noqa: WPS433
    from go2_env import Go2Env  # noqa: WPS433

    _RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = f"mms2_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    log_dir = str(_RUNS_ROOT / run_id)
    exp_name = args.exp_name or run_id

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(exp_name)
    train_cfg["save_interval"] = args.save_interval
    train_cfg["log_interval"] = args.log_interval

    os.makedirs(log_dir, exist_ok=True)
    meta = {
        "stage": "mms_2",
        "pretrain": None,
        "actor_arch": "obs-512-256-128-out",
        "critic_arch": "[512, 256, 128]",
    }

    n_pe = train_cfg["num_steps_per_env"]
    print(
        f"[MMS-2] PPO iter={args.max_iterations}(Training from scratch) | num_steps_per_env={n_pe} | "
        f"num_envs={args.num_envs} | save/log every {args.save_interval} iter"
    )
    print(f"[MMS-2] Network: actor/critic hidden_dims=[512, 256, 128]")
    print(f"[MMS-2] Output: {log_dir}")
    print(f"[TensorBoard] tensorboard --logdir {log_dir}")

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, meta], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    _patch_logger_console_interval(runner.logger, args.log_interval)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
