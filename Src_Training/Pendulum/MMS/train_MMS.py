"""Gymnasium Pendulum-v1 + Stable-Baselines3 SAC (b3_mms/m2).

Network: Strategies pi and Twin-Q(qf) are both obs (or qf side concat(obs,a)) -> 128 -> 128 -> 128 -> output.

Output directory (default is in the directory where this script is located):
  runs/m2_sac_<timestamp>/
    weights/ save checkpoint every approximately 40k cumulative environment timestep (see VecEnv conversion in the script) + final model
    logs/TensorBoard

Run the example:
  python m2_train.py"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--learning-starts", type=int, default=100)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--gradient-steps", type=int, default=1)
    p.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (VecEnv)",
    )
    p.add_argument(
        "--root",
        type=str,
        default="",
        help="Training output root path, default {script directory}/runs/; actual {root}/m2_sac_<timestamp>/",
    )
    p.add_argument(
        "--save-freq",
        type=int,
        default=40_000,
        help=(
            "Checkpoint is saved every accumulated environment timestep "
            "(*_steps in the file name is consistent with this). "
            "SB3 VecEnv internally counts env.step() times, and the script will automatically divide by n_envs."
        ),
    )
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root) if args.root else script_dir / "runs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"m2_sac_{ts}"
    weights_dir = run_dir / "weights"
    logs_dir = run_dir / "logs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.seed)

    policy_kwargs = {
        "net_arch": dict(pi=[128, 128, 128], qf=[128, 128, 128]),
        "share_features_extractor": False,
    }

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
        seed=args.seed,
        tensorboard_log=str(logs_dir),
    )

    # SB3: CheckpointCallback.save_freq = "Number of VecEnv.step() calls". Each call advances n_envs timesteps.
    # If you still write 40000, it will be saved after 40000 calls; 100k/4 will only be called about 25k times and will never be triggered.
    # Official description: save_freq ≈ expected timestep interval // n_envs.
    checkpoint_every = max(args.save_freq // args.n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_every,
        save_path=str(weights_dir),
        name_prefix="sac_m2",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = weights_dir / "sac_m2_final"
    model.save(str(final_path))
    env.close()

    print(f"Training is over. Weight directory: {weights_dir}")
    print(f"Final model: {final_path}.zip")
    print(f"TensorBoard: tensorboard --logdir {logs_dir}")


if __name__ == "__main__":
    main()
