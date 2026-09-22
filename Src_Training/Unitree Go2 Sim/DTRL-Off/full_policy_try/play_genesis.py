#!/usr/bin/env python3
"""
Play FlashSAC checkpoint on Genesis envs (e.g., go2-walk_easy).

Usage:
  python play_genesis.py \
    --checkpoint_path /abs/path/to/step19530 \
    --env_name go2-walk_easy \
    --num_envs 1 \
    --num_episodes 3
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import MutableMapping

import imageio.v2 as imageio
import numpy as np
import torch


def _ensure_uv_runtime(script_path: Path, argv: list[str]) -> None:
    """
    If current interpreter misses hydra/flashsac deps, relaunch via uv project env.
    """
    try:
        import hydra  # noqa: F401
        from omegaconf import OmegaConf  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    if os.environ.get("PLAY_GENESIS_BOOTSTRAPPED") == "1":
        raise RuntimeError("Missing dependencies even after uv bootstrap.")

    flashsac_root = Path("/home/yufeng.yang/codespace/RTSS/FlashSAC")
    uv_bin = str(Path.home() / ".local" / "bin" / "uv")
    cmd = [
        uv_bin,
        "run",
        "--project",
        str(flashsac_root),
        "--extra",
        "genesis",
        "python",
        str(script_path),
        *argv,
    ]
    env = os.environ.copy()
    env["PLAY_GENESIS_BOOTSTRAPPED"] = "1"
    subprocess.run(cmd, check=True, env=env)
    raise SystemExit(0)


def main() -> None:
    _ensure_uv_runtime(Path(__file__).resolve(), sys.argv[1:])
    import hydra
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Play a trained FlashSAC agent in Genesis")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint dir (stepXXXX)")
    parser.add_argument("--env_name", type=str, default="go2-walk_easy", choices=["go2-walk_easy", "go2-walk", "go2-backflip"])
    parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs for playback")
    parser.add_argument("--num_episodes", type=int, default=3, help="Episodes to play")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--disable_render", action="store_true", help="Disable offscreen render calls")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Unitree Go2 Sim/offpolicy/runs/playback",
        help="Directory to save rendered mp4 files",
    )
    parser.add_argument("--fps", type=int, default=25, help="Saved video fps")
    parser.add_argument("--max_steps", type=int, default=4000, help="Safety cap for rollout steps")
    args = parser.parse_args()

    flashsac_root = Path("/home/yufeng.yang/codespace/RTSS/FlashSAC")
    if not flashsac_root.is_dir():
        raise FileNotFoundError(f"FlashSAC root not found: {flashsac_root}")
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    # make `import flash_rl` work when this script is outside FlashSAC root
    if str(flashsac_root) not in sys.path:
        sys.path.insert(0, str(flashsac_root))

    from flash_rl.agents import create_agent
    from flash_rl.envs.genesis import make_genesis_env
    from flash_rl.types import Tensor

    # Config compose with absolute config directory
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    with hydra.initialize_config_dir(version_base=None, config_dir=str(flashsac_root / "configs")):
        cfg = hydra.compose(
            config_name=args.config_name,
            overrides=[
                f"seed={args.seed}",
                "env=genesis",
                f"env.env_name={args.env_name}",
                "agent=flashSAC",
                "num_eval_envs=null",
                "num_record_envs=null",
                # use deterministic actor trunk set in your training path
                "agent.actor_hidden_dims=[512, 256, 128, 128]",
            ],
        )
    OmegaConf.resolve(cfg)

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    env = make_genesis_env(
        env_name=cfg.env.env_name,
        num_envs=args.num_envs,
        rescale_action=cfg.env.rescale_action,
        eval_mode=True,
    )

    observations, env_info = env.reset()
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(str(checkpoint_path))

    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
    episode_returns = np.zeros(args.num_envs, dtype=np.float64)
    completed_episodes = 0
    step_count = 0
    saved_videos: list[Path] = []
    frames: list[np.ndarray] = []
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    video_prefix = f"{args.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    while completed_episodes < args.num_episodes and step_count < args.max_steps:
        actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
        actions = np.asarray(actions)
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
        dones = np.logical_or(terminateds, truncateds)
        episode_returns += rewards

        if not args.disable_render:
            frame_batch = env.render()
            # GenesisVectorEnv render returns shape (1, H, W, C)
            if frame_batch is not None and len(frame_batch) > 0:
                frame = np.asarray(frame_batch[0])
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)

        for i in range(args.num_envs):
            if dones[i]:
                completed_episodes += 1
                print(f"[Episode {completed_episodes}] return={episode_returns[i]:.3f}")
                if not args.disable_render and len(frames) > 0:
                    video_path = save_dir / f"{video_prefix}_ep{completed_episodes:03d}.mp4"
                    imageio.mimwrite(str(video_path), frames, fps=args.fps, quality=8)
                    saved_videos.append(video_path)
                    print(f"[Saved] {video_path}")
                    frames = []
                episode_returns[i] = 0.0
                if completed_episodes >= args.num_episodes:
                    break

        prev_transition = {"next_observation": next_obs}
        step_count += 1

    env.close()
    if step_count >= args.max_steps:
        print(f"[Stop] Reached max_steps={args.max_steps}")
    if saved_videos:
        print("[Playback videos]")
        for p in saved_videos:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
