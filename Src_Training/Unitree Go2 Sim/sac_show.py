"""Visualizing SAC weight inference in Genesis Viewer.

By default, the checkpoint you gave is loaded directly:
  /home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Unitree Go2 Sim/offpolicy/runs/offpolicy/full_20260425_132323_456320/model_1152000.pt

Usage:
  pythonsac_show.py
  python sac_show.py --ckpt /abs/path/to/model_xxx.pt
  python sac_show.py --cpu"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

_LOCOMOTION = Path(__file__).resolve().parent.parent / "Genesis" / "examples" / "locomotion"
if _LOCOMOTION.is_dir():
    sys.path.insert(0, str(_LOCOMOTION))
else:
    raise FileNotFoundError(f"Genesis locomotion not found: {_LOCOMOTION}")

_OFFPOLICY = Path(__file__).resolve().parent / "offpolicy" / "full_policy"
if _OFFPOLICY.is_dir():
    sys.path.insert(0, str(_OFFPOLICY))
else:
    raise FileNotFoundError(f"offpolicy/full_policy not found: {_OFFPOLICY}")

import genesis as gs  # noqa: E402
from go2_env import Go2Env  # noqa: E402
from sac_trainer import GaussianActor  # noqa: E402


DEFAULT_CKPT = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Unitree Go2 Sim/offpolicy/runs/full_20260425_132323_456320/step2_sac/model_1152000.pt"
)


def _load_cfgs(run_dir: Path):
    cfg_path = run_dir / "cfgs.pkl"
    if not cfg_path.is_file() and run_dir.parent.is_dir():
        cfg_path = run_dir.parent / "cfgs.pkl"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"not found {cfg_path}")
    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="SAC checkpoint absolute path (model_xxx.pt)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--max-action",
        type=float,
        default=1.0,
        help="SAC action amplitude consistent with training (full_train defaults to 1.0)",
    )
    args = p.parse_args()

    ckpt = args.ckpt.expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    run_dir = ckpt.parent

    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning", performance_mode=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = _load_cfgs(run_dir)
    reward_cfg = dict(reward_cfg)
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    obs = env.reset()
    obs_t = obs["policy"]
    obs_dim = int(obs_t.shape[-1])
    action_dim = int(env.num_actions)

    hidden = (512, 256, 128, 128)
    try:
        sac_block = train_cfg.get("sac", train_cfg)
        h = sac_block.get("actor", {}).get("hidden_dims")
        if h is None:
            h = train_cfg.get("actor", {}).get("hidden_dims")
        if isinstance(h, (list, tuple)) and len(h) >= 1:
            hidden = tuple(int(x) for x in h)
    except Exception:
        pass

    actor = GaussianActor(obs_dim, action_dim, hidden, max_action=float(args.max_action)).to(gs.device)
    ckpt_dict = torch.load(str(ckpt), map_location=gs.device, weights_only=False)
    actor.load_state_dict(ckpt_dict["actor_state_dict"], strict=True)
    actor.eval()

    print(f"[sac_show] ckpt={ckpt}", flush=True)
    print(f"[sac_show] run_dir={run_dir}", flush=True)
    print(f"[sac_show] hidden={hidden}, max_action={args.max_action}", flush=True)

    with torch.no_grad():
        while True:
            act = actor.deterministic(obs["policy"].to(gs.device))
            obs, _r, _d, _i = env.step(act)


if __name__ == "__main__":
    main()

