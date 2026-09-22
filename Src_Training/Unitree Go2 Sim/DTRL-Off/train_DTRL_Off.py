#!/usr/bin/env python3
"""Go2 three-port joint training: FlashSAC + flashSAC_eenn agent.

Task (e1 / e2 / efull three share the same environment reward and speed instructions):
- Environment: ``go2-walk_easy``, fixed **vx=0.5 m/s, vy=0, yaw=0** to go straight
- Algorithm: FlashSAC SAC; the first 20M env steps only train efull, and then the three steps are combined
- Automatically save ``efull_warmup_env*`` weights when reaching 20M env steps + ``playback/`` next mp4

Run::

    cd ~/codespace/RTSS
    python unitree_go2/DTRL-Off/run_flashsac_joint_go2.py"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

# Fixed speed tasks for three-port joint and full single-port alignment
TASK_CMD_VX = 0.5
TASK_CMD_VY = 0.0
TASK_CMD_YAW = 0.0
ENV_NAME = "go2-walk_easy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 three-port joint (vx=0.5, go straight)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_env_steps", type=str, default="50_000_896")
    parser.add_argument("--num_train_envs", type=int, default=1024)
    parser.add_argument("--joint_warmup_env_steps", type=str, default="20_000_000")
    parser.add_argument("--w_exit1", type=float, default=0.2)
    parser.add_argument("--w_exit2", type=float, default=0.3)
    parser.add_argument("--w_exit3", type=float, default=0.5)
    parser.add_argument("--updates_per_interaction_step", type=int, default=2)
    parser.add_argument("--sample_batch_size", type=int, default=2048)
    parser.add_argument("--buffer_max_length", type=str, default="10_000_000")
    parser.add_argument("--buffer_min_length", type=str, default="100_000")
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--critic_hidden_dim", type=int, default=256)
    parser.add_argument("--critic_num_blocks", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    num_env_steps_int = int(str(args.num_env_steps).replace("_", ""))
    interaction_steps = max(1, num_env_steps_int // int(args.num_train_envs))
    save_ckpt_every = max(1, interaction_steps // 5)
    eval_every = max(1, interaction_steps // 10)

    flashsac_root = Path("/home/yufeng.yang/codespace/RTSS/FlashSAC")
    train_py = flashsac_root / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError(f"Training entrance not found: {train_py}")
    uv_bin = str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv_bin).is_file():
        raise FileNotFoundError(f"uv not found: {uv_bin}")

    runs_root = Path("/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Unitree Go2 Sim/DTRL-Off/runs")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"flashsac_joint_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir_rel = os.path.relpath(run_dir / "checkpoints", start=flashsac_root)
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
        f"env.env_name={ENV_NAME}",
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
        "agent=flashSAC_eenn",
        "--overrides",
        f"agent.joint_warmup_env_steps={args.joint_warmup_env_steps}",
        "--overrides",
        f"agent.w_exit1={args.w_exit1}",
        "--overrides",
        f"agent.w_exit2={args.w_exit2}",
        "--overrides",
        f"agent.w_exit3={args.w_exit3}",
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
        "agent.asymmetric_observation=true",
        "--overrides",
        f"updates_per_interaction_step={args.updates_per_interaction_step}",
        "--overrides",
        f"gamma={args.gamma}",
        "--overrides",
        f"save_checkpoint_per_interaction_step={save_ckpt_every}",
        "--overrides",
        f"evaluation_per_interaction_step={eval_every}",
        "--overrides",
        f"save_path={ckpt_dir_rel}/seed${{seed}}-TIMESTAMP",
        "--overrides",
        f"group_name={run_ts}",
        "--overrides",
        "exp_name=tensorboard",
    ]

    print(" ".join(cmd))
    print(f"[run_dir] {run_dir}")
    print(f"[env] {ENV_NAME}")
    print(
        f"[task] Fixed command vx= shared by three ports{TASK_CMD_VX:+.1f} m/s  "
        f"vy={TASK_CMD_VY:+.1f}  yaw={TASK_CMD_YAW:+.1f}"
    )
    print(f"[joint_warmup_env_steps] {args.joint_warmup_env_steps}")
    warmup_int = int(str(args.joint_warmup_env_steps).replace("_", ""))
    warmup_interaction = max(1, warmup_int // int(args.num_train_envs))
    print(
        f"[efull snapshot] env_step>={warmup_int} Save when "
        f"checkpoints/.../efull_warmup_env<step>/ and playback/efull_warmup_env<step>/*.mp4 "
        f"(about interaction step {warmup_interaction})"
    )
    if args.dry_run:
        return

    env = os.environ.copy()
    env["GO2_WALK_EASY_CMD_VX"] = str(TASK_CMD_VX)
    env["GO2_WALK_EASY_CMD_VY"] = str(TASK_CMD_VY)
    env["GO2_WALK_EASY_CMD_YAW"] = str(TASK_CMD_YAW)
    env["FLASHSAC_ROOT"] = str(flashsac_root)
    env["FLASHSAC_UV_BIN"] = uv_bin
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
