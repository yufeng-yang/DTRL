"""Training PPO full MLP (Stable-Baselines3) on Semicircle **v5**.

Strategy/value network: obs → 256 → 128 → 128 → 64 → 64 → output (pi/vf each independent MLP).
The default 16-way parallel environment; every 50,000 steps, parallel eval 10 games (different seeds) are used to save best_model.
Logs and checkpoints are written to ``<this script directory>/runs/full_ppo_<timestamp>/`` by default.

Please execute it under ``Semicircle-Narrow`` or after configuring ``PYTHONPATH`` before running."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Semicircle_env is located under Semicircle-Narrow/ (the upper-level directory of this script)
_SEMICIRCLE_NARROW = Path(__file__).resolve().parents[1]
if str(_SEMICIRCLE_NARROW) not in sys.path:
    sys.path.insert(0, str(_SEMICIRCLE_NARROW))

import Semicircle_env  # noqa: F401  # register custom env
import safety_gymnasium


class SafetyToGymnasiumWrapper(gym.Wrapper):
    """
    Convert Safety-Gymnasium step API:
        obs, reward, cost, terminated, truncated, info
    to Gymnasium-compatible step API:
        obs, reward, terminated, truncated, info
    """

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["cost"] = float(cost)
        return obs, reward, terminated, truncated, info

    def get_wrapper_attr(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        inner_getter = getattr(self.env, "get_wrapper_attr", None)
        if callable(inner_getter):
            return inner_getter(name)
        return getattr(self.env, name)


class FixedEvalSeedWrapper(gym.Wrapper):
    """For evaluation: Fixed use of specified seed for each reset (facilitates SubprocVecEnv to run multiple seeds in parallel)."""

    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._eval_seed = seed

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=self._eval_seed, options=options)


def make_env(env_id: str, eval_seed: int | None = None):
    def _init():
        env = safety_gymnasium.make(env_id, render_mode=None)
        env = SafetyToGymnasiumWrapper(env)
        env = Monitor(env)
        if eval_seed is not None:
            env = FixedEvalSeedWrapper(env, eval_seed)
        return env

    return _init


_DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent / "runs"


class ParallelSeedEvalCallback(BaseCallback):
    """Multi-seed parallel evaluation: VecEnv runs n rounds at the same time, and best_model is saved according to the average return."""

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        best_model_save_path: str | Path,
        *,
        start_seed: int = 42,
        n_episodes: int = 10,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(int(eval_freq), 1)
        self.best_dir = Path(best_model_save_path)
        self.start_seed = start_seed
        self.n_episodes = n_episodes
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
        seeds = list(range(self.start_seed, self.start_seed + self.n_episodes))
        n_envs = self.eval_env.num_envs
        if n_envs != self.n_episodes:
            raise ValueError(f"evalVecEnv quantity {n_envs} Must be equal to eval episodes {self.n_episodes}")

        obs = self.eval_env.reset()
        episode_rewards = np.zeros(n_envs, dtype=np.float64)
        finished = np.zeros(n_envs, dtype=bool)

        while not finished.all():
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, _ = self.eval_env.step(actions)
            for i in range(n_envs):
                if finished[i]:
                    continue
                episode_rewards[i] += float(rewards[i])
                if dones[i]:
                    finished[i] = True

        mean_ret = float(np.mean(episode_rewards))
        if self.verbose >= 1:
            print(
                f"[Eval] timestep={self.num_timesteps} "
                f"mean_return({self.n_episodes} seeds {seeds[0]}..{seeds[-1]}, parallel)={mean_ret:.6f}"
            )

        if mean_ret > self.best_mean:
            self.best_mean = mean_ret
            self.best_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.best_dir / "best_model"
            self.model.save(str(out_path))
            if self.verbose >= 1:
                print(f"[Eval] new best mean_return={mean_ret:.6f} -> {out_path}.zip")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO (full MLP) on Safety*Semicircle0-v5 with SB3."
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="SafetyPointSemicircle0-v5",
        help="Must be registered Semicircle v5.",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(_DEFAULT_RUNS_ROOT),
        help="Run the root directory; full_ppo_<timestamp>/(checkpoints, tb, best_model, final_model) will be created under it."
        f" default: {_DEFAULT_RUNS_ROOT}",
    )
    parser.add_argument("--save-freq", type=int, default=100_000, help="Checkpoint save frequency in env steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument(
        "--vec-env",
        type=str,
        choices=["dummy", "subproc"],
        default="subproc",
    )
    parser.add_argument("--progress-bar", action="store_true", default=True)
    parser.add_argument("--no-progress-bar", action="store_false", dest="progress_bar")
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Eval frequency in env steps.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Parallel eval episodes per eval.")
    parser.add_argument("--eval-start-seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_root, f"full_ppo_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tb")
    final_model_dir = os.path.join(run_dir, "final_model")
    best_model_dir = os.path.join(run_dir, "best_model")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(final_model_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    env_fns = [make_env(args.env_id) for _ in range(args.n_envs)]
    if args.vec_env == "subproc" and args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    eval_seeds = list(range(args.eval_start_seed, args.eval_start_seed + args.eval_episodes))
    eval_env_fns = [make_env(args.env_id, eval_seed=s) for s in eval_seeds]
    eval_env = SubprocVecEnv(eval_env_fns)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128, 128, 64, 64],
            vf=[256, 128, 128, 64, 64],
        ),
        share_features_extractor=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_dir,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    save_freq_adjusted = max(args.save_freq // max(args.n_envs, 1), 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_adjusted,
        save_path=ckpt_dir,
        name_prefix="ppo_full",
        save_vecnormalize=True,
    )
    eval_callback = ParallelSeedEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        best_model_save_path=best_model_dir,
        start_seed=args.eval_start_seed,
        n_episodes=args.eval_episodes,
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    print(f"[INFO] env_id={args.env_id}")
    print(f"[INFO] total_timesteps={args.total_timesteps}")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] n_envs={args.n_envs}, vec_env={args.vec_env}")
    print(f"[INFO] net_arch pi/vf=[256, 128, 128, 64, 64]")
    print(f"[INFO] checkpoint_save_freq(env_steps)={args.save_freq}")
    print(f"[INFO] eval_freq(env_steps)={args.eval_freq}, eval_episodes={args.eval_episodes} (parallel)")
    print(f"[INFO] eval_seeds={args.eval_start_seed}..{args.eval_start_seed + args.eval_episodes - 1}")
    print(f"[INFO] run_dir={run_dir}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_full",
        progress_bar=args.progress_bar,
    )

    final_model_path = os.path.join(final_model_dir, "ppo_full_final")
    model.save(final_model_path)
    print(f"[INFO] final_model={final_model_path}.zip")

    eval_env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
