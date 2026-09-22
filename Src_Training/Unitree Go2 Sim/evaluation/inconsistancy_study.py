"""Go2 cross-exit action inconsistency (pairwise normalized cross-exit action inconsistency).

For MMS/DTRL_ON/DTRL_OFF:
  1. Each exit rolls out independently, each collects STEPS_PER_EXIT obs before decision-making, and gets S^(i)
  2. S_eval = ⋃_i S^(i) (obs deduplication)
  3. Infer each outlet on obs and calculate the mean of inconsistency

Weights are consistent with mms_eval / dtrl_on_eval / dtrl_off_eval:
  - MMS, DTRL_ON: PPO mean direct env (clip to ±action_range when comparing)
  - DTRL_OFF: FlashSAC standalone tanh output × action_range

Genesis is not suitable for building scenes with multiple processes. Rollout is a single-process serialization; it does not tie the core or time it.

Run (requires Genesis/FlashSAC Python environment installed)::
    python unitree_go2/evaluation/inconsistancy_study.py
    python unitree_go2/evaluation/inconsistancy_study.py --steps-per-exit 500"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

_EVAL = Path(__file__).resolve().parent
_ROOT = _EVAL.parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from time_calculation import (  # noqa: E402
    DEFAULT_ONPOLICY_MODELS,
    STANDALONE_DIR,
    _build_exit_classes,
    _create_env,
    _genesis_teardown,
    _import_flashsac_layer,
    _init_genesis,
    _obs_tensor,
    _pin_commands,
    _standalone_paths,
    load_onpolicy_ppo,
    load_standalone,
)

PolicyKind = Literal["ppo", "standalone"]

N_EXITS = 3
STEPS_PER_EXIT = 500
START_SEED = 42
GS_SEED = 0
OBS_ROUND_DECIMALS = 6
INFER_BATCH = 4096

VX_CMD_LOW = -1.0
VX_CMD_HIGH = 1.0
VY_CMD = 0.0
YAW_CMD = 0.0
VX_CMD_OFF = 0.5

# The weight path is consistent with evaluation/*.py (do not change the path, synchronize here after changing eval or change to import)
from mms_eval import DEFAULT_MODELS as MMS_MODELS  # noqa: E402
from mms_eval import _LOADER_TAG as MMS_LOADER_TAG  # noqa: E402

# DTRL_ON：dtrl_on_eval → time_calculation.DEFAULT_ONPOLICY_MODELS
DTRL_ON_MODELS: dict[str, Path] = dict(DEFAULT_ONPOLICY_MODELS)

# DTRL_OFF：dtrl_off_eval → time_calculation.STANDALONE_DIR

# Unify export order (ef display name; DTRL_ON/OFF weight key is efull)
EXIT_ORDER: tuple[str, ...] = ("e1", "e2", "ef")


@dataclass
class ExitPolicy:
    tag: str
    net: nn.Module
    kind: PolicyKind
    action_range: float
    device: torch.device

    def env_actions(self, obs_batch: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            x = _obs_tensor(obs_batch, self.device)
            raw = self.net(x).detach().cpu().numpy()
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        ar = self.action_range
        if self.kind == "standalone":
            return (raw * ar).astype(np.float32)
        return np.clip(raw, -ar, ar).astype(np.float32)


def _command_tensor(vx: float) -> torch.Tensor:
    import genesis as gs

    return torch.tensor([[vx, VY_CMD, YAW_CMD]], dtype=gs.tc_float, device=gs.device)


def _obs_key(obs: np.ndarray) -> bytes:
    flat = np.asarray(obs, dtype=np.float64).reshape(-1)
    rounded = np.round(flat, OBS_ROUND_DECIMALS)
    return rounded.tobytes()


def _create_mms_env():
    import importlib.util
    import types

    flashsac = (_ROOT.parents[1] / "extra_resources" / "FlashSAC").resolve()
    if str(flashsac) not in sys.path:
        sys.path.insert(0, str(flashsac))
    for pkg in ("flash_rl", "flash_rl.envs", "flash_rl.envs.genesis_envs"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
    path = flashsac / "flash_rl/envs/genesis_envs/go2_walk_easy.py"
    spec = importlib.util.spec_from_file_location(
        "flash_rl.envs.genesis_envs.go2_walk_easy", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flash_rl.envs.genesis_envs.go2_walk_easy"] = mod
    spec.loader.exec_module(mod)
    env_cfg, obs_cfg, reward_cfg, command_cfg = mod.get_cfgs(
        cmd_vy=VY_CMD, cmd_yaw=YAW_CMD
    )
    command_cfg["lin_vel_x_range"] = [VX_CMD_LOW, VX_CMD_HIGH]
    command_cfg["lin_vel_y_range"] = [VY_CMD, VY_CMD]
    return mod.Go2WalkEasyEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )


def _load_mms_policies(device: torch.device, action_range: float) -> list[ExitPolicy]:
    missing = [str(p) for p in MMS_MODELS.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("MMS weight not found:\n" + "\n".join(missing))
    policies: list[ExitPolicy] = []
    for tag in EXIT_ORDER:
        path = MMS_MODELS[tag]
        load_tag = MMS_LOADER_TAG[tag]  # type: ignore[arg-type]
        net = load_onpolicy_ppo(path, load_tag, device)  # type: ignore[arg-type]
        policies.append(ExitPolicy(tag, net, "ppo", action_range, device))
    return policies


def _load_dtrl_on_policies(device: torch.device, action_range: float) -> list[ExitPolicy]:
    policies: list[ExitPolicy] = []
    key_map = {"e1": "e1", "e2": "e2", "ef": "efull"}
    for tag in EXIT_ORDER:
        path = DTRL_ON_MODELS[key_map[tag]]
        if not path.is_file():
            raise FileNotFoundError(path)
        net = load_onpolicy_ppo(path, key_map[tag], device)  # type: ignore[arg-type]
        policies.append(ExitPolicy(tag, net, "ppo", action_range, device))
    return policies


def _load_dtrl_off_policies(device: torch.device, action_range: float) -> list[ExitPolicy]:
    layer_mod = _import_flashsac_layer()
    exit_cls = _build_exit_classes(layer_mod)
    paths = _standalone_paths(STANDALONE_DIR)
    key_map = {"e1": "e1", "e2": "e2", "ef": "efull"}
    policies: list[ExitPolicy] = []
    for tag in EXIT_ORDER:
        p = paths[key_map[tag]]  # type: ignore[index]
        if not p.is_file():
            raise FileNotFoundError(p)
        net = load_standalone(p, device, exit_cls)
        policies.append(ExitPolicy(tag, net, "standalone", action_range, device))
    return policies


def _collect_exit_states(
    env,
    policy: ExitPolicy,
    n_steps: int,
    seed_start: int,
    *,
    random_vx: bool,
) -> list[np.ndarray]:
    """Single outlet rollout independently, collecting n_steps obs before decision-making; automatic reset at the end of the game to continue collecting."""
    import genesis as gs

    states: list[np.ndarray] = []
    ep_idx = 0
    max_steps_per_ep = int(env.max_episode_length) + 50

    while len(states) < n_steps:
        rng = np.random.default_rng(seed_start + ep_idx)
        vx = float(rng.uniform(VX_CMD_LOW, VX_CMD_HIGH)) if random_vx else VX_CMD_OFF
        cmd = _command_tensor(vx)
        _pin_commands(env, cmd)

        obs_buf, _ = env.reset()
        env.commands.copy_(cmd)
        obs = obs_buf.detach().cpu().numpy()
        done = False
        local = 0

        while not done and local < max_steps_per_ep and len(states) < n_steps:
            flat = np.asarray(obs, dtype=np.float32).reshape(-1)
            states.append(flat.copy())

            act = policy.env_actions(flat.reshape(1, -1))[0]
            act_t = torch.as_tensor(act, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
            obs_t, _, dones, _ = env.step(act_t)
            env.commands.copy_(cmd)
            obs = obs_t.detach().cpu().numpy()
            done = bool(dones[0].item())
            local += 1

        ep_idx += 1

    return states


def _union_states(per_exit_states: list[list[np.ndarray]]) -> np.ndarray:
    seen: dict[bytes, np.ndarray] = {}
    for traj in per_exit_states:
        for obs in traj:
            key = _obs_key(obs)
            if key not in seen:
                seen[key] = obs.astype(np.float32)
    if not seen:
        raise ValueError("S_eval is empty")
    return np.stack(list(seen.values()), axis=0)


def _pairwise_inconsistency(actions: np.ndarray, action_denom: float) -> float:
    n = actions.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(actions[i] - actions[j])) / action_denom
            n_pairs += 1
    return total / n_pairs


def _batched_actions(policy: ExitPolicy, obs_batch: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, obs_batch.shape[0], INFER_BATCH):
        end = min(start + INFER_BATCH, obs_batch.shape[0])
        chunks.append(policy.env_actions(obs_batch[start:end]))
    return np.concatenate(chunks, axis=0)


def compute_method_inconsistency(
    method: str,
    policies: list[ExitPolicy],
    env,
    *,
    steps_per_exit: int,
    seed_start: int,
    random_vx: bool,
) -> dict[str, float | int | str | list[str]]:
    n = len(policies)
    assert n == N_EXITS

    per_exit_states: list[list[np.ndarray]] = []
    for pol in policies:
        traj = _collect_exit_states(
            env, pol, steps_per_exit, seed_start, random_vx=random_vx
        )
        per_exit_states.append(traj)

    if len(per_exit_states[0]) < steps_per_exit:
        raise ValueError(
            f"exit {policies[0].tag} Collect only {len(per_exit_states[0])} step, goal {steps_per_exit}"
        )
    act_dim = int(policies[0].env_actions(per_exit_states[0][0].reshape(1, -1)).shape[-1])
    action_denom = float(2.0 * policies[0].action_range * np.sqrt(act_dim))

    s_eval = _union_states(per_exit_states)
    all_actions = [_batched_actions(policies[i], s_eval) for i in range(n)]

    per_state = np.empty(s_eval.shape[0], dtype=np.float64)
    for t in range(s_eval.shape[0]):
        acts = np.stack([all_actions[i][t] for i in range(n)], axis=0)
        per_state[t] = _pairwise_inconsistency(acts, action_denom)

    return {
        "method": method,
        "exits": [p.tag for p in policies],
        "n_exits": n,
        "steps_per_exit": steps_per_exit,
        "states_per_exit": [len(s) for s in per_exit_states],
        "s_eval_size": int(s_eval.shape[0]),
        "action_denom_l2": action_denom,
        "inconsistency_mean": float(per_state.mean()),
        "inconsistency_std": float(per_state.std(ddof=0)),
        "inconsistency_p95": float(np.percentile(per_state, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 cross-exit action inconsistency")
    parser.add_argument(
        "--steps-per-exit",
        type=int,
        default=STEPS_PER_EXIT,
        help=f"The number of steps collected by independent rollout for each outlet (default {STEPS_PER_EXIT}）",
    )
    parser.add_argument("--start-seed", type=int, default=START_SEED)
    args = parser.parse_args()

    steps_per_exit = max(1, int(args.steps_per_exit))
    device = torch.device("cpu")

    print(
        f"Go2 cross-exit inconsistency | per-exit {steps_per_exit} steps "
        f"(seed starting point {args.start_seed}) | Genesis single-process serial | No core tied"
    )
    print("||a_max-a_min||_2 is calculated by env action_range and 12-dimensional action\n")

    method_specs: list[tuple[str, bool, callable]] = [
        ("MMS", True, _create_mms_env),
        ("DTRL_ON", True, _create_mms_env),
        ("DTRL_OFF", False, _create_env),
    ]

    rows: list[dict] = []
    for method, random_vx, env_factory in method_specs:
        _init_genesis(use_cpu=True, gs_seed=GS_SEED)
        env = env_factory()
        action_range = float(env.env_cfg["action_range"])
        try:
            if method == "MMS":
                policies = _load_mms_policies(device, action_range)
            elif method == "DTRL_ON":
                policies = _load_dtrl_on_policies(device, action_range)
            else:
                policies = _load_dtrl_off_policies(device, action_range)

            print(f"{'=' * 70}")
            print(f"method: {method} | export={[p.tag for p in policies]} | action_range={action_range}")
            if random_vx:
                print(f"  rollout: vx∈[{VX_CMD_LOW},{VX_CMD_HIGH}]")
            else:
                print("  rollout: vx=0.5 fixed")

            stats = compute_method_inconsistency(
                method,
                policies,
                env,
                steps_per_exit=steps_per_exit,
                seed_start=args.start_seed,
                random_vx=random_vx,
            )
            rows.append(stats)
            print(f"  Number of rollout states for each exit: {stats['states_per_exit']}")
            print(f"  |S_eval| = {stats['s_eval_size']}")
            print(f"  inconsistency (mean) = {stats['inconsistency_mean']:.6f}")
            print(f"  inconsistency (std)  = {stats['inconsistency_std']:.6f}")
            print(f"  inconsistency (p95)  = {stats['inconsistency_p95']:.6f}")
        finally:
            _genesis_teardown()

    print(f"\n{'=' * 70}")
    print("Summary (in ascending order by inconsistency mean):")
    rows.sort(key=lambda r: float(r["inconsistency_mean"]))
    for r in rows:
        print(
            f"  {r['method']:10s}  I={r['inconsistency_mean']:.6f}  "
            f"|S_eval|={r['s_eval_size']}"
        )


if __name__ == "__main__":
    main()
