"""Load the Go2 PPO checkpoint and evaluate the return and speed error E_v under multiple sets of random speed instructions in parallel.

E_v, E_x, and E_y are all normalized using ||v_cmd||:
E_v = sqrt(mean(||(v_t-v_cmd)/||v_cmd||||²)),
E_x = sqrt(mean((v_x-v_{x,cmd})^2/||v_cmd||²)),
E_y = sqrt(mean((v_y-v_{y,cmd})^2/||v_cmd||²)).

The default **num_envs=10**, within vx∈[-1,1], vy∈[-0.5,0.5], and yaw=0, randomly select 1 set of instructions and lock them throughout.
After removing the trim_frac (default 1%) steps before the start of each game, return and E_v are calculated.

Run::

    python data_show.py
    python data_show.py --cpu
    python data_show.py --model path/to/model_1000.pt"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_LOCOMOTION = _ROOT.parent / "Genesis" / "examples" / "locomotion"
if _LOCOMOTION.is_dir():
    sys.path.insert(0, str(_LOCOMOTION))
else:
    raise FileNotFoundError(f"Genesis locomotion not found: {_LOCOMOTION}")

try:
    if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
        raise ImportError
except (metadata.PackageNotFoundError, ImportError, ValueError) as e:
    raise ImportError("Please install rsl-rl-lib>=5.0.0") from e
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import genesis as gs  # noqa: E402
from go2_env import Go2Env  # noqa: E402

VX_RANGE = (-1.0, 1.0)
VY_RANGE = (-0.5, 0.5)
YAW_CMD = 0.0

# (vx, vy, ret_full, ev, ex, ey, ret_trim, ev, ex, ey, n_steps, n_skip) — The last three errors are whole/trim pairs
EvalRow = tuple[
    float, float,
    float, float, float, float,
    float, float, float, float,
    int, int,
]

_V_CMD_EPS = 1e-8


def _resolve_run_dir_and_ckpt(model_path: Path) -> tuple[Path, int]:
    p = model_path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Weight not found: {p}")
    m = re.fullmatch(r"model_(\d+)\.pt", p.name)
    if not m:
        raise ValueError(f"The weight file name must be model_<iter>.pt, currently: {p.name}")
    return p.parent, int(m.group(1))


def _velocity_errors(
    vels: list[tuple[float, float]],
    v_cmd: tuple[float, float],
) -> tuple[float, float, float]:
    """Return (E_v, E_x, E_y)."""
    if not vels:
        return 0.0, 0.0, 0.0
    v_cmd_arr = np.asarray(v_cmd, dtype=np.float64)
    norm_cmd = float(np.linalg.norm(v_cmd_arr))
    if norm_cmd < _V_CMD_EPS:
        norm_cmd = _V_CMD_EPS
    inv_norm_sq = 1.0 / (norm_cmd * norm_cmd)
    ex_sq: list[float] = []
    ey_sq: list[float] = []
    ev_sq: list[float] = []
    for vx, vy in vels:
        dx = float(vx - v_cmd_arr[0])
        dy = float(vy - v_cmd_arr[1])
        ex_sq.append(dx * dx * inv_norm_sq)
        ey_sq.append(dy * dy * inv_norm_sq)
        ev_sq.append((dx * dx + dy * dy) * inv_norm_sq)
    return (
        float(np.sqrt(np.mean(ev_sq))),
        float(np.sqrt(np.mean(ex_sq))),
        float(np.sqrt(np.mean(ey_sq))),
    )


def _episode_metrics(
    rewards: list[float],
    vels: list[tuple[float, float]],
    v_cmd: tuple[float, float],
    trim_frac: float,
) -> tuple[float, float, float, float, float, float, float, int, int]:
    n = len(rewards)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
    skip = int(np.ceil(n * trim_frac))
    if skip >= n:
        skip = max(n - 1, 0)
    r_trim = rewards[skip:]
    v_trim = vels[skip:]
    ret_full = float(np.sum(rewards))
    ev_full, ex_full, ey_full = _velocity_errors(vels, v_cmd)
    ret_trim = float(np.sum(r_trim)) if r_trim else 0.0
    if v_trim:
        ev_trim, ex_trim, ey_trim = _velocity_errors(v_trim, v_cmd)
    else:
        ev_trim = ex_trim = ey_trim = 0.0
    return ret_full, ev_full, ex_full, ey_full, ret_trim, ev_trim, ex_trim, ey_trim, n, skip


def _genesis_teardown() -> None:
    try:
        gs.destroy()
    except Exception:
        pass


def _sample_commands(
    n: int,
    rng: np.random.Generator,
    *,
    vx_range: tuple[float, float],
    vy_range: tuple[float, float],
) -> np.ndarray:
    """shape (n, 3): vx, vy, yaw."""
    vx = rng.uniform(vx_range[0], vx_range[1], size=n).astype(np.float32)
    vy = rng.uniform(vy_range[0], vy_range[1], size=n).astype(np.float32)
    yaw = np.full(n, YAW_CMD, dtype=np.float32)
    return np.stack([vx, vy, yaw], axis=1)


def _pin_commands(env: Go2Env, fixed: torch.Tensor) -> Callable[..., None]:
    """Fixed the target speed of each env, ignoring the periodic resample within Go2Env."""

    def _pinned_resample(envs_idx=None) -> None:
        if envs_idx is None:
            env.commands.copy_(fixed)
        else:
            idx = envs_idx.nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                env.commands[idx] = fixed[idx]

    env._resample_commands = _pinned_resample  # type: ignore[method-assign]
    return _pinned_resample


def _run_parallel_command_eval(
    *,
    cmd_table: np.ndarray,
    log_dir: Path,
    ckpt: int,
    env_cfg: dict,
    obs_cfg: dict,
    reward_cfg: dict,
    command_cfg: dict,
    train_cfg: dict,
    use_cpu: bool,
    trim_frac: float,
    gs_seed: int,
) -> list[EvalRow]:
    n = int(cmd_table.shape[0])
    _genesis_teardown()
    backend = gs.cpu if use_cpu else gs.gpu
    gs.init(
        backend=backend,
        precision="32",
        logging_level="warning",
        seed=gs_seed,
        performance_mode=True,
    )

    env = Go2Env(
        num_envs=n,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    fixed = torch.as_tensor(cmd_table, dtype=gs.tc_float, device=gs.device)
    _pin_commands(env, fixed)

    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), str(log_dir), device=gs.device)
    runner.load(os.path.join(str(log_dir), f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    reward_lists: list[list[float]] = [[] for _ in range(n)]
    vel_lists: list[list[tuple[float, float]]] = [[] for _ in range(n)]
    finished = [False] * n

    env.reset()
    env.commands.copy_(fixed)
    env._update_observation()
    obs_dict = env.get_observations()
    max_steps = env.max_episode_length + 50

    with torch.no_grad():
        for _ in range(max_steps):
            if all(finished):
                break
            actions = policy(obs_dict)
            obs_dict, rews, dones, _extras = env.step(actions)
            env.commands.copy_(fixed)
            for i in range(n):
                if finished[i]:
                    continue
                reward_lists[i].append(float(rews[i].item()))
                vel_lists[i].append(
                    (float(env.base_lin_vel[i, 0].item()), float(env.base_lin_vel[i, 1].item()))
                )
                d = dones[i]
                if bool(d.item() if hasattr(d, "item") else d):
                    finished[i] = True

    results: list[EvalRow] = []
    for i in range(n):
        vx, vy = float(cmd_table[i, 0]), float(cmd_table[i, 1])
        if not finished[i]:
            _genesis_teardown()
            raise RuntimeError(
                f"env_idx={i} (vx={vx:.3f}, vy={vy:.3f}) does not end the episode within the limited number of steps"
            )
        rf, ev_f, ex_f, ey_f, rt, ev_t, ex_t, ey_t, n_steps, skip = _episode_metrics(
            reward_lists[i], vel_lists[i], (vx, vy), trim_frac
        )
        results.append((vx, vy, rf, ev_f, ex_f, ey_f, rt, ev_t, ex_t, ey_t, n_steps, skip))

    _genesis_teardown()
    return results


def evaluate(
    model_path: Path,
    *,
    num_episodes: int,
    use_cpu: bool,
    trim_frac: float,
    cmd_seed: int | None,
    gs_seed: int,
    vx_range: tuple[float, float],
    vy_range: tuple[float, float],
) -> None:
    log_dir, ckpt = _resolve_run_dir_and_ckpt(model_path)
    cfg_path = log_dir / "cfgs.pkl"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"not found {cfg_path}")

    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    rng = np.random.default_rng(cmd_seed)
    cmd_table = _sample_commands(num_episodes, rng, vx_range=vx_range, vy_range=vy_range)

    print(
        f"[eval] Parallel num_envs={num_episodes}，"
        f"vx∈{vx_range} vy∈{vy_range} yaw={YAW_CMD}，cmd_seed={cmd_seed!r}",
        flush=True,
    )
    for i in range(num_episodes):
        print(
            f"  #{i}: vx={cmd_table[i, 0]:+.3f}  vy={cmd_table[i, 1]:+.3f}",
            flush=True,
        )

    rows = _run_parallel_command_eval(
        cmd_table=cmd_table,
        log_dir=log_dir,
        ckpt=ckpt,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        train_cfg=train_cfg,
        use_cpu=use_cpu,
        trim_frac=trim_frac,
        gs_seed=gs_seed,
    )

    vxs = [r[0] for r in rows]
    vys = [r[1] for r in rows]
    rets_full = [r[2] for r in rows]
    evs_full = [r[3] for r in rows]
    exs_full = [r[4] for r in rows]
    eys_full = [r[5] for r in rows]
    rets_trim = [r[6] for r in rows]
    evs_trim = [r[7] for r in rows]
    exs_trim = [r[8] for r in rows]
    eys_trim = [r[9] for r in rows]
    steps_list = [r[10] for r in rows]
    skips = [r[11] for r in rows]

    rets_full_a = np.asarray(rets_full)
    evs_full_a = np.asarray(evs_full)
    exs_full_a = np.asarray(exs_full)
    eys_full_a = np.asarray(eys_full)
    rets_trim_a = np.asarray(rets_trim)
    evs_trim_a = np.asarray(evs_trim)
    exs_trim_a = np.asarray(exs_trim)
    eys_trim_a = np.asarray(eys_trim)
    v_cmd = np.hypot(np.asarray(vxs), np.asarray(vys))

    print(f"\nLoaded: {model_path}")
    print(f"run_dir: {log_dir}  ckpt={ckpt}")
    print(
        f"Evaluate: {num_episodes} Road parallelism, fixed random instructions,"
        f"backend={'cpu' if use_cpu else 'gpu'}, no rendering"
    )
    print(f"Before each round, remove the {trim_frac*100:.0f}% count after step (ceil(ep_len*frac))")
    print(f"Command vx: {[round(x, 4) for x in vxs]}")
    print(f"Command vy: {[round(x, 4) for x in vys]}")
    print(f"Command |v_cmd|: {[round(x, 4) for x in v_cmd]}")
    print(f"Total number of steps in each round: {steps_list}")
    print(f"Number of steps skipped in each round: {skips}")
    print(f"Each round return (whole round): {rets_full}")
    print(f"Each game returns (go to start {trim_frac*100:.0f}%）: {rets_trim}")
    print(f"Each game E_v (whole game): {[round(x, 4) for x in evs_full]}")
    print(f"Each game E_x (whole game): {[round(x, 4) for x in exs_full]}")
    print(f"Each game E_y (whole game): {[round(x, 4) for x in eys_full]}")
    print(f"Each game E_v (go to start {trim_frac*100:.0f}%）: {[round(x, 4) for x in evs_trim]}")
    print(f"Each game E_x (go to start {trim_frac*100:.0f}%）: {[round(x, 4) for x in exs_trim]}")
    print(f"Each game E_y (go to start {trim_frac*100:.0f}%）: {[round(x, 4) for x in eys_trim]}")
    print()
    n = num_episodes
    print(f"{n} Arithmetic mean return of round (whole round): {rets_full_a.mean():.4f}")
    print(f"{n} Game arithmetic mean E_v (whole game):       {evs_full_a.mean():.4f}  (~{evs_full_a.mean()*100:.1f}%)")
    print(f"{n} Game arithmetic mean E_x (whole game):       {exs_full_a.mean():.4f}  (~{exs_full_a.mean()*100:.1f}%)")
    print(f"{n} Game arithmetic mean E_y (whole game):       {eys_full_a.mean():.4f}  (~{eys_full_a.mean()*100:.1f}%)")
    print(f"{n} Bureau arithmetic mean return (go to start {trim_frac*100:.0f}%）: {rets_trim_a.mean():.4f}")
    print(f"{n} Bureau arithmetic mean E_v (go to start {trim_frac*100:.0f}%）:       {evs_trim_a.mean():.4f}  (~{evs_trim_a.mean()*100:.1f}%)")
    print(f"{n} Bureau arithmetic mean E_x (go to start {trim_frac*100:.0f}%）:       {exs_trim_a.mean():.4f}  (~{exs_trim_a.mean()*100:.1f}%)")
    print(f"{n} Bureau arithmetic mean E_y (go to start {trim_frac*100:.0f}%）:       {eys_trim_a.mean():.4f}  (~{eys_trim_a.mean()*100:.1f}%)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Go2 PPO Renderless Evaluation: Parallel + Random Speed ​​Instructions")
    p.add_argument("--cpu", action="store_true", help="Use CPU backend instead")
    p.add_argument("--episodes", type=int, default=10, help="Number of random instruction groups (number of parallel envs)")
    p.add_argument(
        "--cmd-seed",
        type=int,
        default=None,
        help="Sample the numpy seed of vx/vy; if omitted, it will be different for each run",
    )
    p.add_argument("--gs-seed", type=int, default=0, help="The simulation seed used by gs.init (independent of instruction sampling)")
    p.add_argument(
        "--trim-frac",
        type=float,
        default=0.01,
        help="In each game, the proportion of steps before the start is removed and then counted (default 0.01)",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="model_<iter>.pt absolute path; if omitted, use MODEL_PT in main()",
    )
    return p.parse_args()


def main() -> None:
    MODEL_PT = (
"/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Unitree Go2 Sim/DTRL-On/runs/onpolicy/DTRL-On_full_20260516_092741_449462/DTRL-On_full_8000.pt"
    )
    args = parse_args()
    model_path = Path(args.model or MODEL_PT).expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Weight not found: {model_path}")

    cmd_seed = args.cmd_seed if args.cmd_seed is not None else int.from_bytes(os.urandom(4), "big")

    try:
        evaluate(
            model_path,
            num_episodes=args.episodes,
            use_cpu=args.cpu,
            trim_frac=args.trim_frac,
            cmd_seed=cmd_seed,
            gs_seed=args.gs_seed,
            vx_range=VX_RANGE,
            vy_range=VY_RANGE,
        )
    finally:
        _genesis_teardown()


if __name__ == "__main__":
    main()
