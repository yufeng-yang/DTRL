"""Method 2 (on-policy E2): PPO.

Fixed logic: Actor is obs→128→128→128→action_net (``pi=[128,128,128]``).
Linear: ``latent_pi[0,2,4]`` from SAC_BEST (aligned with first three 128 wide blocks of b1 deep MLP)
Copy into PPO ``policy_net[0,2,4]`` and freeze; only train 128 → action (action_net).
Critic alone obs→128→128→128→V (``vf=[128,128,128]``), share_features_extractor=False.

Run: python onpolicy_e2_train.py"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

# ---------- Fixed path and hyperparameters (no command line options) ----------
SCRIPT_DIR = Path(__file__).resolve().parent
SAC_BEST = (
    SCRIPT_DIR.parent
    / "Full/runs/Full_20260512_044600/weights/Full_final.zip"
)

ENV_ID = "Pendulum-v1"
SEED = 0
TOTAL_TIMESTEPS = 500_000
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
N_ENVS = 4
SAVE_FREQ = 100_000
EVAL_FREQ = 50_000
EVAL_START_SEED = 42
N_EVAL_SEEDS = 100
VERBOSE = 1


class MeanReturnSeedEvalCallback(BaseCallback):
    """Same logic as b1_full_only/train_sac_pendulum_full.py: multiple seeds average return save best."""

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


def transfer_sac_actor_first_three_linear_frozen(sac_path: Path, model: PPO) -> None:
    """SAC ``actor.latent_pi``:Linear(0)+ReLU(1)+Linear(2)+ReLU(3)+Linear(4)+…
    Align with the first three 128 blocks of b1 ``pi=[128,128,128,64,64]``; copy into PPO ``policy_net[0,2,4]`` and freeze.    """
    sac = SAC.load(str(sac_path))
    src_seq = sac.policy.actor.latent_pi
    dst_seq = model.policy.mlp_extractor.policy_net
    for si, di in ((0, 0), (2, 2), (4, 4)):
        src = src_seq[si]
        dst = dst_seq[di]
        if tuple(src.weight.shape) != tuple(dst.weight.shape):
            raise ValueError(
                f"SAC latent_pi[{si}] {tuple(src.weight.shape)} with PPO policy_net[{di}] "
                f"{tuple(dst.weight.shape)} inconsistent"
            )
        dst.load_state_dict(src.state_dict())
        for p in dst.parameters():
            p.requires_grad = False
    del sac


def main() -> None:
    if not SAC_BEST.is_file():
        raise FileNotFoundError(f"SAC weight not found: {SAC_BEST}")

    root = SCRIPT_DIR / "runs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"ppo_e2_sac128x128x128_{ts}"
    weights_dir = run_dir / "weights"
    logs_dir = run_dir / "logs"
    best_dir = weights_dir / "best"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
    eval_env = gym.make(ENV_ID)

    # pi: obs→128→128→128→(action_net); vf: obs→128→128→128→(value header)
    policy_kwargs = {
        "net_arch": dict(pi=[128, 128, 128], vf=[128, 128, 128]),
        "activation_fn": nn.ReLU,
        "share_features_extractor": False,
    }

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=VERBOSE,
        seed=SEED,
        tensorboard_log=str(logs_dir),
    )

    transfer_sac_actor_first_three_linear_frozen(SAC_BEST, model)
    if VERBOSE >= 1:
        print(
            f"[E2] Loaded and frozen latent_pi[0,2,4] → policy_net[0,2,4] from SAC (obs-128-128-128): {SAC_BEST}\n"
            "[E2] Only train action_net (128→out); Critic independent obs→128→128→128→V"
        )

    checkpoint_every = max(SAVE_FREQ // N_ENVS, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_every,
        save_path=str(weights_dir),
        name_prefix="ppo_e2_pendulum",
        save_vecnormalize=False,
    )

    eval_cb = MeanReturnSeedEvalCallback(
        eval_env,
        eval_freq=max(EVAL_FREQ, 1),
        best_model_save_path=str(best_dir),
        start_seed=EVAL_START_SEED,
        n_seeds=N_EVAL_SEEDS,
        verbose=VERBOSE,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([checkpoint_cb, eval_cb]),
        progress_bar=True,
    )

    final_path = weights_dir / "ppo_e2_final"
    model.save(str(final_path))
    env.close()
    eval_env.close()

    print(f"Training is over. Weight directory: {weights_dir}")
    print(f"Evaluate the optimal model: {best_dir / 'best_model.zip'}")
    print(f"TensorBoard: tensorboard --logdir {logs_dir}")


if __name__ == "__main__":
    main()
