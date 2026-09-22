"""Training PPO (Stable-Baselines3) on Semicircle **v5**.

Policy/value network: obs → 256 → output (pi / vf independent MLP, no shared feature extractor).
Default environment ``SafetyPointSemicircle0-v5``; logs and checkpoints are written by default
``<this script directory>/runs/owf_ppo_<timestamp>/`` (i.e. ``b2_owf/runs/...``).

Please execute it under ``Semicircle-Narrow`` or after configuring ``PYTHONPATH`` before running."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
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
        """
        SB3 SubprocVecEnv queries attributes via get_wrapper_attr.
        Safety-Gymnasium's underlying Builder may not implement this API,
        so provide a compatible resolver at wrapper level.
        """
        if hasattr(self, name):
            return getattr(self, name)
        inner_getter = getattr(self.env, "get_wrapper_attr", None)
        if callable(inner_getter):
            return inner_getter(name)
        return getattr(self.env, name)


def make_env(env_id: str):
    def _init():
        env = safety_gymnasium.make(env_id, render_mode=None)
        env = SafetyToGymnasiumWrapper(env)
        env = Monitor(env)
        return env

    return _init


_DEFAULT_RUNS_ROOT = Path(__file__).resolve().parent / "runs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO on Safety*Semicircle0-v5 (default Point) with SB3."
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="SafetyPointSemicircle0-v5",
        help="Must be a registered Semicircle v5, such as SafetyCarSemicircle0-v5, SafetyRacecarSemicircle0-v5.",
    )
    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(_DEFAULT_RUNS_ROOT),
        help="Run the root directory; owf_ppo_<timestamp>/(checkpoints, tb, best_model, final_model) will be created under it."
        f" default: {_DEFAULT_RUNS_ROOT}",
    )
    parser.add_argument("--save-freq", type=int, default=150_000, help="Checkpoint save frequency in env steps.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048, help="PPO rollout length per env before each update.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", help="PPO device: auto/cpu/cuda/cuda:0 ...")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument(
        "--vec-env",
        type=str,
        choices=["dummy", "subproc"],
        default="subproc",
        help="Vectorized env type. Use subproc with n_envs>1 for parallel sampling.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        default=True,
        help="Enable tqdm progress bar (can slightly reduce throughput).",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_false",
        dest="progress_bar",
        help="Disable tqdm progress bar to maximize throughput.",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000, help="Best-model eval frequency in env steps.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Episodes per evaluation.")
    return parser.parse_args()


def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_root, f"SWI_{timestamp}")
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
    eval_env = DummyVecEnv([make_env(args.env_id)])

    policy_kwargs = dict(
        net_arch=dict(
            # obs → 256 → output (the last layer of SB3 is connected by the frame to the action distribution / scalar V)
            pi=[256],
            vf=[256],
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

    # CheckpointCallback save_freq is in callback calls; divide by n_envs to keep env-step-based frequency.
    save_freq_adjusted = max(args.save_freq // max(args.n_envs, 1), 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_adjusted,
        save_path=ckpt_dir,
        name_prefix="ppo_semicircle",
        save_vecnormalize=True,
    )
    eval_freq_adjusted = max(args.eval_freq // max(args.n_envs, 1), 1)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=eval_freq_adjusted,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    print(f"[INFO] env_id={args.env_id}")
    print(f"[INFO] total_timesteps={args.total_timesteps}")
    print(f"[INFO] device={args.device}")
    print(f"[INFO] n_envs={args.n_envs}, vec_env={args.vec_env}")
    print(f"[INFO] n_steps={args.n_steps}, batch_size={args.batch_size}")
    print(f"[INFO] net_arch pi/vf=[256] (obs-256-out, no shared features)")
    print(f"[INFO] checkpoint_save_freq(env_steps)={args.save_freq}")
    print(f"[INFO] checkpoint_save_freq(adjusted_calls)={save_freq_adjusted}")
    print(f"[INFO] best_eval_freq(env_steps)={args.eval_freq}")
    print(f"[INFO] best_eval_freq(adjusted_calls)={eval_freq_adjusted}")
    print(f"[INFO] best_eval_episodes={args.eval_episodes}")
    print(f"[INFO] run_dir={run_dir}")
    print(f"[INFO] checkpoint_dir={ckpt_dir}")
    print(f"[INFO] best_model_dir={best_model_dir}")
    print(f"[INFO] tensorboard_dir={tb_dir}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_semicircle",
        progress_bar=args.progress_bar,
    )

    final_model_path = os.path.join(final_model_dir, "ppo_semicircle_final")
    model.save(final_model_path)
    print(f"[INFO] final_model={final_model_path}.zip")

    eval_env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
