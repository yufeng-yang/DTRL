"""Trained with Stable-Baselines3 SAC on Gymnasium Pendulum-v1,
Policy network structure: Observation -> 256 -> Action (actor / critic are independent MLPs and do not share feature extractors with each other).

Each run will create a new file in this directory:
  runs/owf_<timestamp>/
    weights/ saved .zip model
    logs/ TensorBoard logs

Run the example:
  python train_sac_pendulum.py --total-timesteps 100000"""

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
    p.add_argument("--hidden-dim", type=int, default=128, help="Single layer hidden layer width (obs -> H -> out)")
    p.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (Stable-Baselines3 VecEnv)",
    )
    p.add_argument(
        "--root",
        type=str,
        default="",
        help="Training output root path, defaults to runs/ in the directory where this script is located; the actual directory is {root}/owf_<timestamp>/",
    )
    p.add_argument(
        "--save-freq",
        type=int,
        default=10000,
        help="Save weights to weights/ every how many timesteps",
    )
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root) if args.root else script_dir / "runs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"owf_{ts}"
    weights_dir = run_dir / "weights"
    logs_dir = run_dir / "logs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.seed)

    policy_kwargs = {
        # Single layer 256: pi / qf each have one MLP and no shared layers with each other (SB3 SAC does not allow ac to share layers other than features)
        "net_arch": dict(pi=[args.hidden_dim], qf=[args.hidden_dim]),
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

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq, 1),
        save_path=str(weights_dir),
        name_prefix="sac_pendulum",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = weights_dir / "sac_pendulum_final"
    model.save(str(final_path))
    env.close()

    print(f"Training is over. Weight directory: {weights_dir}")
    print(f"TensorBoard: tensorboard --logdir {logs_dir}")


if __name__ == "__main__":
    main()
