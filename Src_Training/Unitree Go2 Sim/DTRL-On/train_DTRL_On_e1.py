"""Go2 PPO Stage-E1: Freeze full obs→512, and only retrain 512→action (E1 exit obs-512-out).

Load actor ``mlp.0`` from full checkpoint (corresponding to ``hidden_dims[0]=512`` of ``full_train``) and freeze;
This script actor ``hidden_dims=[512]`` only trains ``mlp.2`` (512→12) and ``distribution``.

Default pretrained weights::

    DTRL-On/runs/onpolicy/DTRL-On_full_20260516_092741_449462/DTRL-On_full_8000.pt

Training: 3000 iter; ``save_interval=300``; ``log_interval=300`` (console prints every 300 iter).

Run::

    python unitree_go2/DTRL-On/ppo_e1_train.py"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import torch

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.") from e
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger

_SCRIPT_DIR = Path(__file__).resolve().parent
_GENESIS_LOCOMOTION_REL = Path("Genesis") / "examples" / "locomotion"


def _resolve_locomotion_dir(explicit: str | None = None) -> Path:
    """Locate Genesis/examples/locomotion (search upward, compatible with different warehouse layouts).

    Priority: ``--genesis-locomotion`` / ``GENESIS_LOCOMOTION`` / ``RTSS_ROOT`` /
    Look up ``.../Genesis/examples/locomotion`` from this script directory and cwd.    """
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
        "Genesis locomotion not found (requires go2_env.py). Tried:\n  "
        + "\n  ".join(tried[:12])
        + ("\n  ..." if len(tried) > 12 else "")
        + "\nYou can set the environment variable GENESIS_LOCOMOTION or RTSS_ROOT, or use --genesis-locomotion."
    )


def _setup_go2_import(locomotion_dir: Path) -> None:
    loc = str(locomotion_dir.resolve())
    if loc not in sys.path:
        sys.path.insert(0, loc)

_DEFAULT_PRETRAIN = (
    _SCRIPT_DIR
    / "runs/onpolicy/DTRL-On_full_20260516_092741_449462/DTRL-On_full_8000.pt"
)
_RUNS_ROOT = _SCRIPT_DIR / "runs" / "onpolicy"


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
            "hidden_dims": [512],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 300,
        "log_interval": 300,
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


def load_and_freeze_actor_obs512(runner: OnPolicyRunner, pretrain_path: Path) -> None:
    """Copy mlp.0 (obs→512) from full actor to E1 actor and freeze."""
    ckpt = torch.load(str(pretrain_path), map_location=runner.device, weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise KeyError(f"{pretrain_path} actor_state_dict missing")
    full_sd = ckpt["actor_state_dict"]
    actor = runner.alg.actor
    trunk = actor.mlp[0]
    expected = trunk.state_dict()
    loaded = {}
    for name in expected:
        key = f"mlp.0.{name}"
        if key not in full_sd:
            raise KeyError(f"Pretrained weights are missing {key}")
        if tuple(full_sd[key].shape) != tuple(expected[name].shape):
            raise ValueError(
                f"{key} Shape mismatch: full {tuple(full_sd[key].shape)} vs E1 {tuple(expected[name].shape)}"
            )
        loaded[name] = full_sd[key].clone()
    trunk.load_state_dict(loaded)
    for p in trunk.parameters():
        p.requires_grad = False


def _trainable_actor_param_names(actor) -> list[str]:
    return [n for n, p in actor.named_parameters() if p.requires_grad]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Go2 PPO E1: freeze full obs→512, retrain action head")
    p.add_argument("-e", "--exp_name", type=str, default=None)
    p.add_argument("-B", "--num_envs", type=int, default=4096)
    p.add_argument("--max_iterations", type=int, default=3000)
    p.add_argument("--save-interval", type=int, default=300)
    p.add_argument("--log-interval", type=int, default=300)
    p.add_argument(
        "--pretrain",
        type=str,
        default=str(_DEFAULT_PRETRAIN),
        help="full PPO model_<iter>.pt, used to load and freeze actor mlp.0",
    )
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
    print(f"[E1] Genesis locomotion: {locomotion_dir}")

    import genesis as gs  # noqa: WPS433
    from go2_env import Go2Env  # noqa: WPS433

    pretrain_path = Path(args.pretrain).expanduser().resolve()
    if not pretrain_path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrain_path}")

    _RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = f"ppo_e1_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    log_dir = str(_RUNS_ROOT / run_id)
    exp_name = args.exp_name or run_id

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(exp_name)
    train_cfg["save_interval"] = args.save_interval
    train_cfg["log_interval"] = args.log_interval

    os.makedirs(log_dir, exist_ok=True)
    meta = {
        "stage": "ppo_e1",
        "pretrain_pt": str(pretrain_path),
        "actor_arch": "obs-512-out (frozen mlp.0)",
        "critic_arch": "[512]",
    }

    n_pe = train_cfg["num_steps_per_env"]
    print(
        f"[E1] PPO iter={args.max_iterations} | num_steps_per_env={n_pe} | num_envs={args.num_envs} | "
        f"save/log every {args.save_interval} iter"
    )
    print(f"[E1] Pre-training (freeze trunk): {pretrain_path}")
    print(f"[E1] Output: {log_dir}")

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
    load_and_freeze_actor_obs512(runner, pretrain_path)

    trainable = _trainable_actor_param_names(runner.alg.actor)
    print(f"[E1] actor trainable parameters ({len(trainable)}):")
    for n in trainable:
        print(f"  - {n}")

    _patch_logger_console_interval(runner.logger, args.log_interval)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
