"""Trained with Stable-Baselines3 SAC on Gymnasium Pendulum-v1,
Strategy/Q network structure: Observation -> 128 -> 128 -> 128 -> 64 -> 64 -> Output (pi and qf are independent MLPs).

Each run will create a new file in this directory:
  runs/sac_<timestamp>/
    weights/regular checkpoint (default cumulative environment timestep every 40k)
    weights/best/ According to the average return of "100 consecutive seeds starting from seed=42", save best_model.zip (default is evaluated every 10k timestep)
    logs/ TensorBoard logs

Run the example:
  python train_sac_pendulum.py --total-timesteps 100000"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env


class MeanReturnSeedEvalCallback(BaseCallback):
    """Every eval_freq "accumulated environment timestep", use the deterministic strategy on a single environment
    Run n_seeds consecutively from start_seed, compare the average return with the historical optimal, and save best_model.zip if it is better.    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int,
        best_model_save_path: str | Path,
        *,
        start_seed: int = 42,
        n_seeds: int = 100,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(int(eval_freq), 1)
        self.best_dir = Path(best_model_save_path)
        self.start_seed = start_seed
        self.n_seeds = n_seeds
        self.best_mean = -np.inf
        self.next_eval_at = self.eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_eval_at:
            return True
        while self.num_timesteps >= self.next_eval_at:
            self.next_eval_at += self.eval_freq
            self._eval_and_maybe_save()
        return True

    def _eval_and_maybe_save(self) -> None:
        rets: list[float] = []
        for ep_seed in range(self.start_seed, self.start_seed + self.n_seeds):
            obs, _ = self.eval_env.reset(seed=ep_seed)
            ep_ret = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                ep_ret += float(reward)
            rets.append(ep_ret)

        mean_ret = float(np.mean(rets))
        if self.verbose >= 1:
            print(
                f"[Eval] timestep={self.num_timesteps} "
                f"mean_return({self.n_seeds} seeds {self.start_seed}..{self.start_seed + self.n_seeds - 1})={mean_ret:.6f}"
            )

        if mean_ret > self.best_mean:
            self.best_mean = mean_ret
            self.best_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.best_dir / "best_model"
            self.model.save(str(out_path))
            if self.verbose >= 1:
                print(f"[Eval] new best mean_return={mean_ret:.6f} -> {out_path}.zip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=150_000)
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
        help="Number of parallel environments (Stable-Baselines3 VecEnv)",
    )
    p.add_argument(
        "--root",
        type=str,
        default="",
        help="Training output root path, defaults to runs/ in the directory where this script is located; the actual directory is {root}/sac_<timestamp>/",
    )
    p.add_argument(
        "--save-freq",
        type=int,
        default=40_000,
        help="Save the checkpoint every accumulated environment timestep; under VecEnv the script divides by n_envs before passing to SB3.",
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Every cumulative environment timestep does a 100-seed evaluation and possibly updates best.",
    )
    p.add_argument(
        "--eval-start-seed",
        type=int,
        default=42,
        help="The episode seed to start from when evaluating (n-eval-seeds in a row).",
    )
    p.add_argument(
        "--n-eval-seeds",
        type=int,
        default=100,
        help="The number of seeds run continuously for each evaluation (the average return is used to select the best).",
    )
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root) if args.root else script_dir / "runs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"sac_{ts}"
    weights_dir = run_dir / "weights"
    logs_dir = run_dir / "logs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    best_dir = weights_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=args.seed)
    eval_env = gym.make(args.env_id)

    policy_kwargs = {
        "net_arch": dict(pi=[128, 128, 128, 64, 64], qf=[128, 128, 128, 64, 64]),
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

    checkpoint_every = max(args.save_freq // args.n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_every,
        save_path=str(weights_dir),
        name_prefix="sac_pendulum",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_cb = MeanReturnSeedEvalCallback(
        eval_env,
        eval_freq=max(args.eval_freq, 1),
        best_model_save_path=str(best_dir),
        start_seed=args.eval_start_seed,
        n_seeds=args.n_eval_seeds,
        verbose=args.verbose,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([checkpoint_cb, eval_cb]),
        progress_bar=True,
    )

    final_path = weights_dir / "sac_pendulum_final"
    model.save(str(final_path))
    env.close()
    eval_env.close()

    print(f"Training is over. Weight directory: {weights_dir}")
    print(f"Evaluate the optimal model (by {args.n_eval_seeds}-seed average return): {best_dir / 'best_model.zip'}")
    print(f"TensorBoard: tensorboard --logdir {logs_dir}")


if __name__ == "__main__":
    main()
