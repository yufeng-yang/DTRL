"""Compare the actions and rewards of render and data_show with the same weight and the same instruction vx=0.5 (single env, no viewer)."""

from __future__ import annotations

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_LOC = _ROOT.parent / "Genesis" / "examples" / "locomotion"
sys.path.insert(0, str(_LOC))

import genesis as gs
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

MODEL = _ROOT / "DTRL-On/runs/onpolicy/DTRL-On_full_20260516_092741_449462/DTRL-On_full_8000.pt"
VX, VY, YAW = 0.5, 0.0, 0.0
GS_SEED = 0
MAX_STEPS = 1000  # Long enough, usually ends at max_episode_length


def _pin_commands(env: Go2Env, fixed: torch.Tensor) -> None:
    def _pinned(envs_idx=None) -> None:
        if envs_idx is None:
            env.commands.copy_(fixed)
        else:
            env.commands[envs_idx] = fixed[envs_idx]

    env._resample_commands = _pinned


def _load_policy(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, log_dir: Path, ckpt: int):
    env = Go2Env(1, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False)
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), str(log_dir), device=gs.device)
    runner.load(str(log_dir / f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)
    return env, policy


def _run_episode(label: str, *, clear_reward_scales: bool) -> dict:
    log_dir = MODEL.parent
    ckpt = 8000
    with open(log_dir / "cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)

    env_cfg = dict(env_cfg)
    reward_cfg = dict(reward_cfg)
    if clear_reward_scales:
        reward_cfg["reward_scales"] = {}  # render.py behavior

    gs.destroy()
    gs.init(backend=gs.cpu, precision="32", logging_level="warning", seed=GS_SEED, performance_mode=True)

    env, policy = _load_policy(env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, log_dir, ckpt)
    fixed = torch.tensor([[VX, VY, YAW]], dtype=gs.tc_float, device=gs.device)
    _pin_commands(env, fixed)

    default_dof = env.default_dof_pos.detach().cpu().numpy().reshape(-1)

    actions_log: list[np.ndarray] = []
    rewards_log: list[float] = []
    obs_after_reset = None

    env.reset()
    env.commands.copy_(fixed)
    with torch.no_grad():
        obs_dict = env.get_observations()
        obs_after_reset = obs_dict["policy"].detach().cpu().numpy().reshape(-1).copy()

        for step in range(MAX_STEPS):
            act = policy(obs_dict).detach().cpu().numpy().reshape(-1).copy()
            obs_dict, rews, dones, _ = env.step(
                torch.as_tensor(act, dtype=gs.tc_float, device=gs.device).unsqueeze(0)
            )
            env.commands.copy_(fixed)
            actions_log.append(act)
            rewards_log.append(float(rews[0].item()))
            if bool(dones[0].item()):
                break

    gs.destroy()
    return {
        "label": label,
        "clear_reward_scales": clear_reward_scales,
        "default_dof_pos": default_dof,
        "n_steps": len(actions_log),
        "actions": actions_log,
        "rewards": rewards_log,
        "obs_reset": obs_after_reset,
        "return_sum": float(np.sum(rewards_log)),
    }


def _compare(a: dict, b: dict) -> None:
    print(f"\n{'='*72}")
    print("Initial default_dof_pos (should be the same, from the same cfgs.pkl)")
    print(f"  render path: {a['default_dof_pos']}")
    print(f"  data_show path: {b['default_dof_pos']}")
    print(f"  max |diff|: {np.max(np.abs(a['default_dof_pos'] - b['default_dof_pos'])):.6e}")

    print(f"\nreset after obs['policy'] (should be the same, same as gs_seed={GS_SEED}）")
    print(f"  max |diff|: {np.max(np.abs(a['obs_reset'] - b['obs_reset'])):.6e}")

    n = min(a["n_steps"], b["n_steps"])
    print(f"\nNumber of steps: render mode={a['n_steps']}  data_show mode={b['n_steps']}  before comparison {n} step")

    act_diffs = []
    rew_diffs = []
    for i in range(n):
        da = np.max(np.abs(a["actions"][i] - b["actions"][i]))
        dr = abs(a["rewards"][i] - b["rewards"][i])
        act_diffs.append(da)
        rew_diffs.append(dr)

    act_diffs = np.asarray(act_diffs)
    rew_diffs = np.asarray(rew_diffs)

    print("\n--- action policy output ---")
    print(f"  Step by step max|Δaction|: min={act_diffs.min():.3e} max={act_diffs.max():.3e} mean={act_diffs.mean():.3e}")
    n_act_same = int(np.sum(act_diffs < 1e-5))
    print(f"  |Δaction|<1e-5 steps: {n_act_same}/{n}")

    first_mismatch = int(np.argmax(act_diffs >= 1e-5)) if np.any(act_diffs >= 1e-5) else -1
    if first_mismatch >= 0:
        i = first_mismatch
        print(f"  The first action is inconsistent step={i}:")
        print(f"    render:    {a['actions'][i]}")
        print(f"    data_show: {b['actions'][i]}")
        print(f"    |diff|:    {act_diffs[i]}")

    print("\n--- Gradually reward ---")
    print(f"  render total return:    {a['return_sum']:.6f}  (reward_scales cleared)")
    print(f"  data_show total return: {b['return_sum']:.6f}  (full reward_scales)")
    print(f"  Step by step |Δreward|: min={rew_diffs.min():.6f} max={rew_diffs.max():.6f}")
    print("  Comparison of rewards in the first 10 steps:")
    print(f"    {'step':>4}  {'render':>12}  {'data_show':>12}  {'|diff|':>12}")
    for i in range(min(10, n)):
        print(f"    {i:4d}  {a['rewards'][i]:12.6f}  {b['rewards'][i]:12.6f}  {rew_diffs[i]:12.6f}")


def main() -> None:
    print(f"Model: {MODEL}")
    print(f"Command: vx={VX}, vy={VY}, yaw={YAW}")
    print(f"gs_seed={GS_SEED}, num_envs=1")

    a = _run_episode("render_style", clear_reward_scales=True)
    b = _run_episode("data_show_style", clear_reward_scales=False)
    _compare(a, b)


if __name__ == "__main__":
    main()
