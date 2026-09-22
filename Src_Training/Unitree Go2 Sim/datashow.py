"""FlashSAC EENN: 1 round for each of the three ports to evaluate return and average speed (vx=0.5 m/s straight, no rendering).

Default weight step48825. Run::

    python unitree_go2/datashow.py
    python unitree_go2/datashow.py --cpu"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Literal

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    _ROOT
    / "DTRL-Off/runs/DTRL-Off_20260517_091613/checkpoints"
    / "seed42-0517-091635/step48825"
)
_FLASHSAC = _ROOT.parents[1] / "extra_resources" / "FlashSAC"

VX_CMD = 0.5
VY_CMD = 0.0
YAW_CMD = 0.0

ExitName = Literal["e1", "e2", "efull"]
EXITS: tuple[ExitName, ...] = ("e1", "e2", "efull")


def _genesis_teardown() -> None:
    try:
        import genesis as gs

        gs.destroy()
    except Exception:
        pass


def _resolve_actor_pt(path: Path) -> Path:
    p = path.expanduser().resolve()
    if p.is_file() and p.suffix == ".pt":
        return p
    direct = p / "actor.pt"
    if direct.is_file():
        return direct
    raise FileNotFoundError(f"actor.pt not found: {p}")


def _import_flashsac_eenn_actor_class():
    flashsac = _FLASHSAC.resolve()
    if str(flashsac) not in sys.path:
        sys.path.insert(0, str(flashsac))

    def _load_py_module(dotted: str, file_path: Path):
        if dotted in sys.modules:
            return sys.modules[dotted]
        spec = importlib.util.spec_from_file_location(dotted, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module: {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    for pkg in (
        "flash_rl",
        "flash_rl.agents",
        "flash_rl.agents.flashSAC",
        "flash_rl.agents.utils",
    ):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    _load_py_module(
        "flash_rl.agents.utils.distribution",
        flashsac / "flash_rl/agents/utils/distribution.py",
    )
    _load_py_module(
        "flash_rl.agents.flashSAC.layer",
        flashsac / "flash_rl/agents/flashSAC/layer.py",
    )
    eenn_mod = _load_py_module(
        "flash_rl.agents.flashSAC.eenn_actor",
        flashsac / "flash_rl/agents/flashSAC/eenn_actor.py",
    )
    return eenn_mod.FlashSACEENNActor


def _load_actor(actor_pt: Path, device: torch.device):
    FlashSACEENNActor = _import_flashsac_eenn_actor_class()
    ckpt = torch.load(actor_pt, map_location=device, weights_only=False)
    raw = ckpt.get("network_state_dict", ckpt.get("actor", ckpt))
    if not isinstance(raw, dict):
        raise TypeError(f"Unable to parse checkpoint: {actor_pt}")
    state = {
        (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v for k, v in raw.items()
    }
    obs_dim, act_dim = 45, 12
    for key, tensor in state.items():
        if key == "backbone.0.weight":
            obs_dim = int(tensor.shape[1])
        if key == "exit1.mean_w.w.weight":
            act_dim = int(tensor.shape[0])
    actor = FlashSACEENNActor(obs_dim, act_dim).to(device)
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def _import_go2_walk_easy_mod():
    flashsac = _FLASHSAC.resolve()
    if str(flashsac) not in sys.path:
        sys.path.insert(0, str(flashsac))
    for pkg in ("flash_rl", "flash_rl.envs", "flash_rl.envs.genesis_envs"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)
    path = flashsac / "flash_rl/envs/genesis_envs/go2_walk_easy.py"
    spec = importlib.util.spec_from_file_location("flash_rl.envs.genesis_envs.go2_walk_easy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flash_rl.envs.genesis_envs.go2_walk_easy"] = mod
    spec.loader.exec_module(mod)
    return mod


def _create_env():
    mod = _import_go2_walk_easy_mod()
    env_cfg, obs_cfg, reward_cfg, command_cfg = mod.get_cfgs(
        cmd_vx=VX_CMD, cmd_vy=VY_CMD, cmd_yaw=YAW_CMD
    )
    return mod.Go2WalkEasyEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )


def _pin_commands(env, fixed: torch.Tensor) -> None:
    def _pinned(envs_idx=None) -> None:
        if envs_idx is None:
            env.commands.copy_(fixed)
        else:
            idx = envs_idx.nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                env.commands[idx] = fixed[idx]

    env._resample_commands = _pinned  # type: ignore[method-assign]


@torch.no_grad()
def _exit_action(actor, obs: np.ndarray, exit_name: ExitName, device: torch.device) -> np.ndarray:
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if exit_name == "e1":
        h0, _, _ = actor._features(x)
        mean, _ = actor.exit1.get_mean_and_std(h0, training=False)
    elif exit_name == "e2":
        _, h2, _ = actor._features(x)
        mean, _ = actor.exit2.get_mean_and_std(h2, training=False)
    else:
        mean, _ = actor.get_mean_and_std(x, training=False)
    return torch.tanh(mean).cpu().numpy()


def _run_one_episode(
    actor,
    exit_name: ExitName,
    *,
    device: torch.device,
    use_cpu: bool,
    gs_seed: int,
) -> tuple[float, float, float, int]:
    """Return (return, average vx, average vy, number of steps)."""
    import genesis as gs

    _genesis_teardown()
    backend = gs.cpu if use_cpu else gs.gpu
    gs.init(
        backend=backend,
        precision="32",
        logging_level="warning",
        seed=gs_seed,
        performance_mode=True,
    )

    env = _create_env()
    action_range = float(env.env_cfg["action_range"])
    fixed = torch.as_tensor([[VX_CMD, VY_CMD, YAW_CMD]], dtype=gs.tc_float, device=gs.device)
    _pin_commands(env, fixed)
    obs_buf, _ = env.reset()
    env.commands.copy_(fixed)
    obs = obs_buf.detach().cpu().numpy()
    reward_sum = 0.0
    vx_list: list[float] = []
    vy_list: list[float] = []
    max_steps = int(env.max_episode_length) + 50

    for step_i in range(max_steps):
        act = _exit_action(actor, obs, exit_name, device) * action_range
        act_t = torch.as_tensor(act, dtype=gs.tc_float, device=gs.device)
        obs_t, rews, dones, _ = env.step(act_t)
        env.commands.copy_(fixed)
        obs = obs_t.detach().cpu().numpy()
        reward_sum += float(rews[0].item())
        vx_list.append(float(env.base_lin_vel[0, 0].item()))
        vy_list.append(float(env.base_lin_vel[0, 1].item()))

        if bool(dones[0].item()):
            _genesis_teardown()
            return reward_sum, float(np.mean(vx_list)), float(np.mean(vy_list)), step_i + 1

    _genesis_teardown()
    raise RuntimeError(f"{exit_name} Exceed {max_steps} step not finished")


def run(*, checkpoint: Path, use_cpu: bool, gs_seed: int) -> None:
    device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    actor_pt = _resolve_actor_pt(checkpoint)
    actor = _load_actor(actor_pt, device)

    print(f"[datashow] Weight: {actor_pt}")
    print(f"[datashow] command vx={VX_CMD:+.3f} vy={VY_CMD:+.3f} yaw={YAW_CMD:+.3f}  1 episode per bite")
    print()

    rows: list[tuple[str, float, float, float, int]] = []
    for exit_name in EXITS:
        ret, mean_vx, mean_vy, n_steps = _run_one_episode(
            actor, exit_name, device=device, use_cpu=use_cpu, gs_seed=gs_seed
        )
        rows.append((exit_name, ret, mean_vx, mean_vy, n_steps))
        print(
            f"  [{exit_name}] return={ret:8.3f}  "
            f"average speed vx={mean_vx:+.4f} vy={mean_vy:+.4f}  "
            f"|v|={np.hypot(mean_vx, mean_vy):.4f}  steps={n_steps}"
        )

    print()
    print("=" * 72)
    print(f"{'exit':<8} {'return':>10} {'averagevx':>10} {'average vy':>10} {'|v|':>10} {'number of steps':>8}")
    print("-" * 72)
    for exit_name, ret, mean_vx, mean_vy, n_steps in rows:
        print(
            f"{exit_name:<8} {ret:10.3f} {mean_vx:+10.4f} {mean_vy:+10.4f} "
            f"{np.hypot(mean_vx, mean_vy):10.4f} {n_steps:8d}"
        )
    print("=" * 72)
    print(f"Target command: vx={VX_CMD:+.3f} m/s")


def main() -> None:
    p = argparse.ArgumentParser(description="Three mouth return / average speed (0.5 m/s)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gs-seed", type=int, default=0)
    args = p.parse_args()

    ckpt = DEFAULT_CHECKPOINT
    if not ckpt.exists() and not (ckpt / "actor.pt").exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    try:
        run(checkpoint=ckpt, use_cpu=args.cpu, gs_seed=args.gs_seed)
    finally:
        _genesis_teardown()


if __name__ == "__main__":
    main()
