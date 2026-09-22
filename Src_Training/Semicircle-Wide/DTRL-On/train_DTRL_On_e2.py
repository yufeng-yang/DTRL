"""Semicircle v6 PPO Stage-E2: Freeze the obs→256→128→128 of full PPO, and only retrain the action header.

Load ``policy_net[0,2,4]`` from ``train_ppo_full`` weights (``pi=[256,128,128,64,64]``)
(obs→256→128→128) and freeze; this script ``pi=[256,128,128]``, only trains ``action_net`` (128→out)
and independent Critic ``vf=[256,128,128]``.

Run: conda activate sb3sg && python dtrl_ppo_e2.py"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

_SCRIPT_DIR = Path(__file__).resolve().parent
_SEMICIRCLE_BOARD = _SCRIPT_DIR.parent
for _p in (_SCRIPT_DIR, _SEMICIRCLE_BOARD):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import Semicircle_env  # noqa: F401
import safety_gymnasium

from train_DTRL_On_full import FixedEvalSeedWrapper, ParallelSeedEvalCallback, SafetyToGymnasiumWrapper


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

_DEFAULT_PRETRAIN_PPO = (
    _DEFAULT_RUNS_ROOT
    / "DTRL-On_full_20260517_145457/checkpoints/ppo_full_1000000_steps.zip"
)

_TRAIN_DEVICE = "cpu"


def transfer_ppo_full_trunk_frozen(pretrain_path: Path, model: PPO) -> None:
    """Copy obs→256→128→128 (Linear 0,2,4) from policy_net of full PPO and freeze; only action_net is trainable."""
    src = PPO.load(str(pretrain_path))
    src_seq = src.policy.mlp_extractor.policy_net
    dst_seq = model.policy.mlp_extractor.policy_net
    pairs = ((0, 0), (2, 2), (4, 4))
    for si, di in pairs:
        src_layer = src_seq[si]
        dst_layer = dst_seq[di]
        if tuple(src_layer.weight.shape) != tuple(dst_layer.weight.shape):
            raise ValueError(
                f"full PPO policy_net[{si}] {tuple(src_layer.weight.shape)} and "
                f"E2 policy_net[{di}] {tuple(dst_layer.weight.shape)} inconsistent"
            )
        dst_layer.load_state_dict(src_layer.state_dict())
        for p in dst_layer.parameters():
            p.requires_grad = False
    del src


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO E2 stage on Semicircle v6 (frozen 256-128-128 trunk).")
    p.add_argument("--env-id", type=str, default="SafetyPointSemicircle0-v6")
    p.add_argument("--pretrain-ppo", type=str, default=str(_DEFAULT_PRETRAIN_PPO))
    p.add_argument("--total-timesteps", type=int, default=1_500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs-root", type=str, default=str(_DEFAULT_RUNS_ROOT))
    p.add_argument("--save-freq", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-start-seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--vec-env", type=str, choices=["dummy", "subproc"], default="subproc")
    p.add_argument("--progress-bar", action="store_true", default=True)
    p.add_argument("--no-progress-bar", action="store_false", dest="progress_bar")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pretrain_path = Path(args.pretrain_ppo).expanduser().resolve()
    if not pretrain_path.is_file():
        raise FileNotFoundError(f"Full PPO pretrained weights not found: {pretrain_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs_root, f"ppo_e2_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    tb_dir = os.path.join(run_dir, "tb")
    final_model_dir = os.path.join(run_dir, "final_model")
    best_model_dir = os.path.join(run_dir, "best_model")
    for d in (ckpt_dir, tb_dir, final_model_dir, best_model_dir):
        os.makedirs(d, exist_ok=True)

    env_fns = [make_env(args.env_id) for _ in range(args.n_envs)]
    if args.vec_env == "subproc" and args.n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    eval_seeds = list(range(args.eval_start_seed, args.eval_start_seed + args.eval_episodes))
    eval_env_fns = [make_env(args.env_id, eval_seed=s) for s in eval_seeds]
    eval_env = SubprocVecEnv(eval_env_fns)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 128], vf=[256, 128, 128]),
        activation_fn=nn.ReLU,
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
        device=_TRAIN_DEVICE,
        verbose=1,
    )

    transfer_ppo_full_trunk_frozen(pretrain_path, model)
    trainable = [n for n, p in model.policy.named_parameters() if p.requires_grad]
    print(f"[E2] Loaded and frozen full PPO trunk (policy_net 0,2,4): {pretrain_path}")
    print(f"[E2] Trainable parameters: {trainable}")

    save_freq_adjusted = max(args.save_freq // max(args.n_envs, 1), 1)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_adjusted,
        save_path=ckpt_dir,
        name_prefix="ppo_e2",
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
    print(f"[INFO] net_arch pi=[256,128,128] (trunk frozen), vf=[256,128,128] (trainable)")
    print(f"[INFO] n_envs={args.n_envs}, device={_TRAIN_DEVICE}")
    print(f"[INFO] run_dir={run_dir}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_e2",
        progress_bar=args.progress_bar,
    )

    final_model_path = os.path.join(final_model_dir, "ppo_e2_final")
    model.save(final_model_path)
    print(f"[INFO] final_model={final_model_path}.zip")

    eval_env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
