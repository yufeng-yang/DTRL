#!/usr/bin/env python3
"""Start FlashSAC training (Genesis Go2) under unitree_go2.

Description:
- This script only passes Hydra overrides and does not change the training cycle.
- Actor supports explicit hidden_dims (strictly set to 512,256,128,128).
- Critic keeps the original FlashSAC parameters by default and can be changed individually as needed."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="go2-walk_easy",
        choices=["go2-walk", "go2-walk_easy", "go2-backflip"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_env_steps", type=str, default="50_000_896")
    parser.add_argument("--num_train_envs", type=int, default=1024)

    # Actor / Critic can be configured independently
    parser.add_argument(
        "--actor_hidden_dims",
        type=str,
        default="512,256,128,128",
        help="Comma separated; if it is an empty string, it will fall back to the original actor_hidden_dim + actor_num_blocks",
    )
    parser.add_argument("--actor_hidden_dim", type=int, default=128)
    parser.add_argument("--actor_num_blocks", type=int, default=2)
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--critic_num_blocks", type=int, default=2)

    # Common training items
    parser.add_argument("--updates_per_interaction_step", type=int, default=2)
    parser.add_argument("--sample_batch_size", type=int, default=2048)
    parser.add_argument("--buffer_max_length", type=str, default="10_000_000")
    parser.add_argument("--buffer_min_length", type=str, default="100_000")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Save checkpoint every 20% of total interaction steps.
    # interaction_steps = num_env_steps / num_train_envs
    num_env_steps_int = int(str(args.num_env_steps).replace("_", ""))
    interaction_steps = max(1, num_env_steps_int // int(args.num_train_envs))
    save_ckpt_every = max(1, interaction_steps // 5)

    flashsac_root = Path("/home/yufeng.yang/codespace/RTSS/FlashSAC")
    train_py = flashsac_root / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"Training entrance not found: {train_py}")
    # Follow the installation process you gave: uv is installed in ~/.local/bin/uv
    uv_bin = str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv_bin).is_file():
        raise FileNotFoundError(f"uv not found: {uv_bin}")

    # Unified output root: unitree_go2/offpolicy/runs/<timestamp>/
    runs_root = Path("/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Unitree Go2 Sim/offpolicy/runs")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir_rel = os.path.relpath(run_dir / "checkpoints", start=flashsac_root)
    # Hydra initialize() requires a relative path; here it is fixed to pass the "configs" of the same level as train.py
    # (More stable than cross-directory ../../.. to avoid being parsed into /home/FlashSAC/configs)
    config_path_rel = "configs"

    cmd = [
        uv_bin,
        "run",
        "--project",
        str(flashsac_root),
        "--extra",
        "genesis",
        "python",
        str(train_py),
        "--config_path",
        config_path_rel,
        "--config_name",
        "flashSAC_base",
        "--overrides",
        f"seed={args.seed}",
        "--overrides",
        "env=genesis",
        "--overrides",
        f"env.env_name={args.env_name}",
        "--overrides",
        f"num_env_steps={args.num_env_steps}",
        "--overrides",
        f"num_train_envs={args.num_train_envs}",
        "--overrides",
        "num_eval_envs=null",
        "--overrides",
        "num_record_envs=null",
        "--overrides",
        "num_eval_episodes=1024",
        "--overrides",
        "num_record_episodes=1",
        "--overrides",
        "agent=flashSAC",
        "--overrides",
        f"agent.actor_hidden_dim={args.actor_hidden_dim}",
        "--overrides",
        f"agent.actor_num_blocks={args.actor_num_blocks}",
        "--overrides",
        f"agent.critic_hidden_dim={args.critic_hidden_dim}",
        "--overrides",
        f"agent.critic_num_blocks={args.critic_num_blocks}",
        "--overrides",
        f"agent.buffer_max_length={args.buffer_max_length}",
        "--overrides",
        f"agent.buffer_min_length={args.buffer_min_length}",
        "--overrides",
        "agent.buffer_device_type=cuda",
        "--overrides",
        f"agent.sample_batch_size={args.sample_batch_size}",
        "--overrides",
        "agent.use_amp=true",
        "--overrides",
        f"updates_per_interaction_step={args.updates_per_interaction_step}",
        "--overrides",
        "agent.asymmetric_observation=true",
        "--overrides",
        f"gamma={args.gamma}",
        "--overrides",
        f"n_step={args.n_step}",
        "--overrides",
        f"save_checkpoint_per_interaction_step={save_ckpt_every}",
        "--overrides",
        f"save_path={ckpt_dir_rel}/seed${{seed}}-TIMESTAMP",
        "--overrides",
        f"group_name={run_ts}",
        "--overrides",
        "exp_name=tensorboard",
    ]

    actor_hidden_dims = [x.strip() for x in args.actor_hidden_dims.split(",") if x.strip()]
    if actor_hidden_dims:
        actor_hidden_dims_str = ", ".join(str(int(x)) for x in actor_hidden_dims)
        cmd.extend(["--overrides", f"agent.actor_hidden_dims=[{actor_hidden_dims_str}]"])
    else:
        cmd.extend(["--overrides", "agent.actor_hidden_dims=null"])

    print(" ".join(cmd))
    print(f"[run_dir] {run_dir}")
    print(f"[tensorboard_dir] {run_dir / 'runs'}")
    print(f"[checkpoint_root] {run_dir / 'checkpoints'}")
    if args.dry_run:
        return

    # Write down the key environment variables according to the environment installation process you gave.
    env = os.environ.copy()
    cuda_home = "/usr/local/cuda"
    home = str(Path.home())
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{home}/.local/bin:{cuda_home}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"{cuda_home}/lib64:"
        f"{home}/.mujoco/mujoco210/bin:"
        f"{env.get('LD_LIBRARY_PATH', '')}:/usr/lib/nvidia"
    )
    env["MUJOCO_GL"] = "egl"
    env["MUJOCO_EGL_DEVICE_ID"] = "0"
    env["MKL_SERVICE_FORCE_INTEL"] = "0"

    subprocess.run(cmd, cwd=str(run_dir), env=env, check=True)


if __name__ == "__main__":
    main()
