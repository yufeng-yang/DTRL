"""Go2 FlashSAC EENN three-port random switching self evaluation (standalone independent weight).

Loaded from evaluation/standalone_models/step48825/{e1,e2,efull}.pt,
Switch uniformly and randomly between e1/e2/efull every N steps. Task: go2_walk_easy, vx=0.5 m/s go straight.
action_range is taken from environment env_cfg (3.0), not from ckpt default.

Run::

    python unitree_go2/evaluation/self_eval.py
    python unitree_go2/evaluation/self_eval.py --cpu
    python unitree_go2/evaluation/self_eval.py --n-episodes 10"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

_EVAL = Path(__file__).resolve().parent
_ROOT = _EVAL.parent
_FLASHSAC = _ROOT.parents[1] / "extra_resources" / "FlashSAC"

if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from time_calculation import (  # noqa: E402
    STANDALONE_DIR,
    _build_exit_classes,
    _import_flashsac_layer,
    _standalone_paths,
    load_standalone,
)

VX_CMD = 0.5
VY_CMD = 0.0
YAW_CMD = 0.0
V_CMD = (VX_CMD, VY_CMD)

ExitName = Literal["e1", "e2", "efull"]
EXIT_TAGS: tuple[ExitName, ...] = ("e1", "e2", "efull")

GS_SEED = 0
N_EPISODES = 10
SWITCH_INTERVAL = 10
EV_THRESHOLD = 0.11
_V_CMD_EPS = 1e-8


@dataclass
class EpisodeStats:
    ep: int
    return_: float
    n_steps: int
    survived: bool
    e_v: float
    e_x: float
    e_y: float
    success: bool
    exit_counts: dict[str, int] = field(default_factory=dict)


def _genesis_teardown() -> None:
    try:
        import genesis as gs

        gs.destroy()
    except Exception:
        pass


def _obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


def _load_standalone_exits(
    standalone_dir: Path,
    device: torch.device,
) -> dict[ExitName, nn.Module]:
    layer_mod = _import_flashsac_layer()
    exit_cls = _build_exit_classes(layer_mod)
    paths = _standalone_paths(standalone_dir)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Standalone weight not found:\n" + "\n".join(missing)
        )
    nets: dict[ExitName, nn.Module] = {}
    for tag in EXIT_TAGS:
        net, _ = load_standalone(paths[tag], device, exit_cls)
        nets[tag] = net
    return nets


@torch.inference_mode()
def _exit_action(
    nets: dict[ExitName, nn.Module],
    obs: np.ndarray,
    exit_name: ExitName,
    device: torch.device,
) -> np.ndarray:
    act = nets[exit_name](_obs_tensor(obs, device)).detach().cpu().numpy()
    if act.ndim == 1:
        act = act.reshape(1, -1)
    return act


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


def _velocity_errors(vels: list[tuple[float, float]]) -> tuple[float, float, float]:
    if not vels:
        return 0.0, 0.0, 0.0
    v_cmd_arr = np.asarray(V_CMD, dtype=np.float64)
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


def _init_genesis(use_cpu: bool, gs_seed: int) -> None:
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


def _run_episode(
    env,
    nets: dict[ExitName, nn.Module],
    *,
    action_range: float,
    fixed: torch.Tensor,
    device: torch.device,
    rng: np.random.Generator,
    switch_interval: int,
    ep_idx: int,
) -> EpisodeStats:
    import genesis as gs

    obs_buf, _ = env.reset()
    env.commands.copy_(fixed)
    obs = obs_buf.detach().cpu().numpy()

    ep_ret = 0.0
    step = 0
    vels: list[tuple[float, float]] = []
    exit_counts: dict[str, int] = {tag: 0 for tag in EXIT_TAGS}
    active: ExitName = EXIT_TAGS[int(rng.integers(0, len(EXIT_TAGS)))]
    done = False
    survived = False

    max_steps = int(env.max_episode_length) + 50
    while not done and step < max_steps:
        if step % switch_interval == 0:
            active = EXIT_TAGS[int(rng.integers(0, len(EXIT_TAGS)))]

        act = _exit_action(nets, obs, active, device) * action_range
        exit_counts[active] += 1

        act_t = torch.as_tensor(act, dtype=gs.tc_float, device=gs.device)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(0)
        obs_t, rews, dones, extras = env.step(act_t)
        env.commands.copy_(fixed)
        obs = obs_t.detach().cpu().numpy()
        ep_ret += float(rews[0].item())
        vels.append(
            (float(env.base_lin_vel[0, 0].item()), float(env.base_lin_vel[0, 1].item()))
        )
        done = bool(dones[0].item())
        step += 1

        if done:
            time_outs = extras.get("time_outs")
            if time_outs is not None:
                survived = float(time_outs[0].item()) > 0.5
            else:
                survived = step >= int(env.max_episode_length)

    e_v, e_x, e_y = _velocity_errors(vels)
    success = survived and e_v <= EV_THRESHOLD and e_x <= EV_THRESHOLD and e_y <= EV_THRESHOLD
    return EpisodeStats(
        ep=ep_idx,
        return_=ep_ret,
        n_steps=step,
        survived=survived,
        e_v=e_v,
        e_x=e_x,
        e_y=e_y,
        success=success,
        exit_counts=exit_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 three ports randomly switch self eval (standalone)")
    parser.add_argument("--cpu", action="store_true", help="Genesis uses CPU (default GPU)")
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--rng-seed", type=int, default=42, help="Export randomly switches RNG seeds")
    parser.add_argument(
        "--standalone-dir",
        type=Path,
        default=STANDALONE_DIR,
        help="Standalone model directory (default evaluation/standalone_models/step48825)",
    )
    parser.add_argument(
        "--switch-interval",
        type=int,
        default=SWITCH_INTERVAL,
        help=f"Randomly switch exits every few steps (default {SWITCH_INTERVAL}）",
    )
    args = parser.parse_args()

    standalone_dir = args.standalone_dir.expanduser().resolve()
    if not standalone_dir.is_dir():
        raise SystemExit(f"Standalone directory not found: {standalone_dir}")

    use_cpu = args.cpu
    device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    n_episodes = int(args.n_episodes)
    rng = np.random.default_rng(int(args.rng_seed))
    switch_interval = max(1, int(args.switch_interval))

    nets = _load_standalone_exits(standalone_dir, device)
    model_paths = _standalone_paths(standalone_dir)

    print(f"Weight: standalone {standalone_dir}")
    for tag in EXIT_TAGS:
        print(f"  [{tag}] {model_paths[tag]}")
    print(
        f"Task: go2_walk_easy vx={VX_CMD} | every {switch_interval} step randomly switches three ports"
    )
    print(f"equipment: {device} | episodes={n_episodes} | rng_seed={args.rng_seed}")

    _init_genesis(use_cpu, GS_SEED)
    import genesis as gs

    env = _create_env()
    action_range = float(env.env_cfg["action_range"])
    fixed = torch.tensor([[VX_CMD, VY_CMD, YAW_CMD]], dtype=gs.tc_float, device=gs.device)
    _pin_commands(env, fixed)
    print(f"action_range={action_range}(from env.env_cfg)")

    results: list[EpisodeStats] = []
    try:
        for ep in range(n_episodes):
            st = _run_episode(
                env,
                nets,
                action_range=action_range,
                fixed=fixed,
                device=device,
                rng=rng,
                switch_interval=switch_interval,
                ep_idx=ep,
            )
            results.append(st)
            print(
                f"  ep {ep + 1}/{n_episodes} return={st.return_:.4f} steps={st.n_steps} "
                f"survived={st.survived} E_v={st.e_v:.4f} E_x={st.e_x:.4f} E_y={st.e_y:.4f} "
                f"success={st.success} exits={st.exit_counts}"
            )
    finally:
        _genesis_teardown()

    rets = np.array([r.return_ for r in results], dtype=np.float64)
    evs = np.array([r.e_v for r in results], dtype=np.float64)
    success_rate = float(np.mean([r.success for r in results]))
    total_exits = {tag: sum(r.exit_counts[tag] for r in results) for tag in EXIT_TAGS}

    print(f"\n{'=' * 70}")
    print(f"comprehensive ({n_episodes} episodes) — three-port random switching (standalone):")
    print(f"  mean return   = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  mean E_v      = {evs.mean():.4f} ± {evs.std(ddof=0):.4f}")
    print(f"  success rate  = {success_rate:.4f} ({sum(r.success for r in results)}/{n_episodes})")
    print(f"  total exits   = {total_exits}")


if __name__ == "__main__":
    main()
