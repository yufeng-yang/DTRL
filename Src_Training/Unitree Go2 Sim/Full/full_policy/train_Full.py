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

import genesis as gs

# go2_env is located in the Genesis repository examples/locomotion, regardless of which directory it is started from.
_LOCOMOTION = Path(__file__).resolve().parents[3] / "Genesis" / "examples" / "locomotion"
if _LOCOMOTION.is_dir():
    sys.path.insert(0, str(_LOCOMOTION))
else:
    raise FileNotFoundError(
        f"not found { _LOCOMOTION }, please confirm that this script is still under RTSS/unitree_go2/onpolicy/full_policy/ and that the Genesis submodule exists."
    )

from go2_env import Go2Env


def get_train_cfg(exp_name):
    train_cfg_dict = {
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
            "hidden_dims": [512, 256, 128, 128, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128, 128, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 1000,
        "log_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def _patch_logger_console_interval(logger: Logger, interval: int) -> None:
    """rsl-rl prints every iter by default; it changes to complete log/print every interval times (including TensorBoard)."""
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


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
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
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--exp_name",
        type=str,
        default=None,
        help="Only the run_name of train_cfg is written; the saving directory name is fixed to full_<timestamp>, which is the same as the directory name by default.",
    )
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=None, help="Console/TB print interval (default 100)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # Separate directory for each run: onpolicy/runs/onpolicy/full_YYYYMMDD_HHMMSS_ffffff/
    _runs_root = Path(__file__).resolve().parents[1] / "runs" / "onpolicy"
    _runs_root.mkdir(parents=True, exist_ok=True)
    run_id = f"full_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    log_dir = str(_runs_root / run_id)
    exp_name = args.exp_name if args.exp_name is not None else run_id
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(exp_name)
    log_interval = args.log_interval if args.log_interval is not None else int(train_cfg.get("log_interval", 100))
    train_cfg["log_interval"] = log_interval

    os.makedirs(log_dir, exist_ok=True)

    n_pe = train_cfg["num_steps_per_env"]
    total_transitions = args.max_iterations * n_pe * args.num_envs
    print(
        f"[Training scale] Number of PPO update iterations: {args.max_iterations} | "
        f"Number of rollout steps per environment per round num_steps_per_env: {n_pe} | "
        f"Number of parallel environments: {args.num_envs} | "
        f"Cumulative number of sampled (env, step) transitions: {total_transitions}"
    )
    print(f"[Output directory] checkpoint and TensorBoard: {log_dir}")
    print(f"[TensorBoard] tensorboard --logdir {log_dir}")
    print(f"[Log] Console with TensorBoard per {log_interval} iter records once")

    with open(f"{log_dir}/cfgs.pkl", "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    _patch_logger_console_interval(runner.logger, log_interval)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""# Instructions: vx∈[-1,1] vy∈[-0.5,0.5] yaw=0 (same random range as train_swi)
#Default 10000 iter; save_interval=1000; log_interval=100 (the console prints every 100 iter)
# python unitree_go2/DTRL-On/full_policy/full_train.py"""
