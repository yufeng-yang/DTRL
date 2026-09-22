"""Semicircle narrow quick self evaluation: parallel rollout, counting returns and collisions.

-PPO E1 checkpoint
- N_EPISODES Bureau (SubprocVecEnv 16-way parallel batching) + SB3 predict
- Impact: Any step with info[cost]>0 counts 1 time (maximum 1 time per step)

Run: python self_eval.py"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

_NARROW = Path(__file__).resolve().parent.parent
_ONPOLICY = _NARROW / "DTRL-On"
sys.path.insert(0, str(_NARROW))
sys.path.insert(0, str(_ONPOLICY))

import Semicircle_env  # noqa: F401
import safety_gymnasium  # noqa: E402

from train_DTRL_On_full import FixedEvalSeedWrapper, SafetyToGymnasiumWrapper  # noqa: E402

MODEL_ZIP = Path(
"/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Semicircle-Narrow/SWI/runs/SWI_20260516_000551/checkpoints/SWI_1500000_steps.zip"
)
ENV_ID = "SafetyPointSemicircle0-v5"
N_EPISODES = 48
N_PARALLEL = 16
START_SEED = 42


@dataclass
class EpResult:
    seed: int
    return_: float
    collision_steps: int

    @property
    def had_collision(self) -> bool:
        return self.collision_steps > 0


def _make_env_fn(env_id: str, eval_seed: int):
    def _init():
        env = safety_gymnasium.make(env_id, render_mode=None)
        env = SafetyToGymnasiumWrapper(env)
        env = Monitor(env)
        env = FixedEvalSeedWrapper(env, eval_seed)
        return env

    return _init


def _env_id_from_model(model: PPO, fallback: str) -> str:
    if model.env is not None and hasattr(model.env, "spec") and model.env.spec is not None:
        return str(model.env.spec.id)
    return fallback


def _step_cost(info: dict | None) -> float:
    if not info:
        return 0.0
    return float(info.get("cost", 0.0))


def _rollout_batch(
    model: PPO,
    env_id: str,
    seeds: list[int],
) -> list[EpResult]:
    n = len(seeds)
    vec_env = SubprocVecEnv([_make_env_fn(env_id, s) for s in seeds])
    try:
        obs = vec_env.reset()
        episode_rewards = np.zeros(n, dtype=np.float64)
        collision_steps = np.zeros(n, dtype=np.int64)
        finished = np.zeros(n, dtype=bool)

        while not finished.all():
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(actions)
            for i in range(n):
                if finished[i]:
                    continue
                episode_rewards[i] += float(rewards[i])
                if _step_cost(infos[i]) > 0:
                    collision_steps[i] += 1
                if dones[i]:
                    finished[i] = True

        return [
            EpResult(
                seed=seeds[i],
                return_=float(episode_rewards[i]),
                collision_steps=int(collision_steps[i]),
            )
            for i in range(n)
        ]
    finally:
        vec_env.close()


def _parallel_rollout(
    model: PPO,
    env_id: str,
    *,
    n_episodes: int,
    start_seed: int,
    n_parallel: int = N_PARALLEL,
) -> list[EpResult]:
    all_seeds = list(range(start_seed, start_seed + n_episodes))
    results: list[EpResult] = []
    for offset in range(0, n_episodes, n_parallel):
        batch_seeds = all_seeds[offset : offset + n_parallel]
        results.extend(_rollout_batch(model, env_id, batch_seeds))
    return results


def _print_summary(model_path: Path, results: list[EpResult]) -> None:
    rets = np.asarray([r.return_ for r in results], dtype=np.float64)
    cols = np.asarray([r.collision_steps for r in results], dtype=np.int64)
    hits = np.asarray([r.had_collision for r in results], dtype=bool)
    seeds = [r.seed for r in results]

    per_ret = ", ".join(f"s{s}={r.return_:.4f}" for s, r in zip(seeds, results))
    per_col = ", ".join(f"s{s}={r.collision_steps}" for s, r in zip(seeds, results))

    print(f"\n{'=' * 70}")
    print(f"[ppo_e1] {model_path}")
    print(f"  episodes={len(results)}  seeds={seeds[0]}..{seeds[-1]}")
    print(f"  mean return = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  min={rets.min():.4f}  max={rets.max():.4f}  median={float(np.median(rets)):.4f}")
    print(f"  per-ep return: {per_ret}")
    print(
        f"  Impact rate (proportion of rounds with cost>0 for any step): "
        f"{hits.mean() * 100:.1f}% ({int(hits.sum())}/{len(results)})"
    )
    print(f"  Average number of collision steps: {cols.mean():.2f} ± {cols.std(ddof=0):.2f}")
    print(f"  Total collision steps: {int(cols.sum())}")
    print(f"  Number of collision steps per-ep: {per_col}")


def main() -> None:
    if not MODEL_ZIP.is_file():
        raise SystemExit(f"Not found: {MODEL_ZIP}")

    print(
        f"Semicircle narrow self eval | {N_EPISODES} episodes | "
        f"seeds {START_SEED}..{START_SEED + N_EPISODES - 1}"
    )

    model = PPO.load(str(MODEL_ZIP), device="cpu")
    env_id = _env_id_from_model(model, ENV_ID)

    print(
        f"\n>>> env={env_id} | {N_EPISODES} eps | "
        f"{N_PARALLEL} Road parallel batching | predict(deterministic=True)"
    )
    results = _parallel_rollout(
        model, env_id, n_episodes=N_EPISODES, start_seed=START_SEED
    )
    del model
    _print_summary(MODEL_ZIP, results)


if __name__ == "__main__":
    main()
