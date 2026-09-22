"""Go2 three-port reasoning time-consuming distribution.

Default on-policy PPO (efull/e1/e2)::
    ef: DTRL-On/.../full_.../DTRL-On_full_8000.pt
    e1: DTRL-On/.../ppo_e1_.../DTRL-On_e1_2999.pt
    e2: DTRL-On/.../ppo_e2_.../DTRL-On_e2_2999.pt

Optional FlashSAC standalone::
    evaluation/standalone_models/step48825/{e1,e2,efull}.pt

Task: go2_walk_easy, vx=0.5 m/s go straight.
Each outlet: warmup 1 episode, officially 10 episodes; the first 1% of each game is removed from statistics.
Default CPU; bound to CPU core 1, torch single thread.

Run::
    python unitree_go2/evaluation/time_calculation.py
    python unitree_go2/evaluation/time_calculation.py --backend standalone
    python unitree_go2/evaluation/time_calculation.py --gpu"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import types
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

_EVAL = Path(__file__).resolve().parent
_ROOT = _EVAL.parent
_FLASHSAC = _ROOT.parents[1] / "extra_resources" / "FlashSAC"
_ONPOLICY = _ROOT / "DTRL-On/runs/onpolicy"

STANDALONE_DIR = _EVAL / "standalone_models" / "step48825"

DEFAULT_ONPOLICY_MODELS: dict[str, Path] = {
    "efull": _ONPOLICY / "DTRL-On_full_20260516_092741_449462/DTRL-On_full_8000.pt",
    "e1": _ONPOLICY / "DTRL-On_e1_20260516_102657_632627/DTRL-On_e1_2999.pt",
    "e2": _ONPOLICY / "DTRL-On_e2_20260516_104308_450115/DTRL-On_e2_2999.pt",
}

VX_CMD = 0.5
VY_CMD = 0.0
YAW_CMD = 0.0

ExitName = Literal["e1", "e2", "efull"]
BackendName = Literal["onpolicy", "standalone"]

_EXITS: list[tuple[ExitName, int, str]] = [
    ("e1", 1, "obs→512→out (PPO E1)"),
    ("e2", 2, "obs→512→256→128→out (PPO E2)"),
    ("efull", 3, "obs→512→256→128→128→128→out (PPO full)"),
]

_STANDALONE_ARCH: dict[ExitName, str] = {
    "e1": "obs→512→exit1",
    "e2": "obs→512→256→128→exit2",
    "efull": "obs→512→256→128→128→128→exit3",
}

N_EPISODES = 10
TRIM_FRAC = 0.01
GS_SEED = 0
CPU_CORE = 1
_PCTS = (1, 5, 10, 90, 95, 97, 99)


def _setup_cpu(*, quiet: bool = False) -> None:
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {CPU_CORE})
        if not quiet:
            print(f"[cpu] bind core {CPU_CORE}，affinity={sorted(os.sched_getaffinity(0))}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _genesis_teardown() -> None:
    try:
        import genesis as gs

        gs.destroy()
    except Exception:
        pass


# --- on-policy PPO actors (RSL-RL MLPModel, ELU, deterministic mean) ---


class PpoE1Actor(nn.Module):
    def __init__(self, obs_dim: int = 45, act_dim: int = 12) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class PpoE2Actor(nn.Module):
    def __init__(self, obs_dim: int = 45, act_dim: int = 12) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class PpoEfullActor(nn.Module):
    def __init__(self, obs_dim: int = 45, act_dim: int = 12) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def _ppo_actor_cls(tag: ExitName) -> type[nn.Module]:
    if tag == "e1":
        return PpoE1Actor
    if tag == "e2":
        return PpoE2Actor
    return PpoEfullActor


def load_onpolicy_ppo(
    model_path: Path,
    tag: ExitName,
    device: torch.device,
) -> nn.Module:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "actor_state_dict" not in ckpt:
        raise TypeError(f"Unable to parse PPO checkpoint: {model_path}")
    raw = ckpt["actor_state_dict"]
    mlp_sd = {k: v for k, v in raw.items() if k.startswith("mlp.")}
    if not mlp_sd:
        raise KeyError(f"{model_path} Missing actor mlp weights")

    obs_dim, act_dim = 45, 12
    w0 = mlp_sd.get("mlp.0.weight")
    w_out_keys = ("mlp.10.weight", "mlp.6.weight", "mlp.2.weight")
    if w0 is not None:
        obs_dim = int(w0.shape[1])
    for key in w_out_keys:
        w = mlp_sd.get(key)
        if w is not None:
            act_dim = int(w.shape[0])
            break

    net = _ppo_actor_cls(tag)(obs_dim, act_dim).to(device)
    net.load_state_dict(mlp_sd, strict=True)
    return net.eval()


# --- FlashSAC standalone (optional) ---


def _import_flashsac_layer():
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

    for pkg in ("flash_rl", "flash_rl.agents", "flash_rl.agents.flashSAC", "flash_rl.agents.utils"):
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    _load_py_module(
        "flash_rl.agents.utils.distribution",
        flashsac / "flash_rl/agents/utils/distribution.py",
    )
    return _load_py_module(
        "flash_rl.agents.flashSAC.layer",
        flashsac / "flash_rl/agents/flashSAC/layer.py",
    )


def _build_exit_classes(layer_mod) -> dict[ExitName, type[nn.Module]]:
    UnitRMSNorm = layer_mod.UnitRMSNorm
    NormalTanhPolicy = layer_mod.NormalTanhPolicy

    class Exit1Standalone(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(obs_dim, 512), nn.ReLU(inplace=True))
            self.backbone_norm = UnitRMSNorm(512)
            self.exit1 = NormalTanhPolicy(512, act_dim)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            h0 = self.backbone_norm(self.backbone(obs))
            mean, _ = self.exit1.get_mean_and_std(h0, training=False)
            return torch.tanh(mean)

    class Exit2Standalone(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(obs_dim, 512), nn.ReLU(inplace=True))
            self.backbone_norm = UnitRMSNorm(512)
            self.fc256 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
            self.norm256 = UnitRMSNorm(256)
            self.fc128 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
            self.norm128 = UnitRMSNorm(128)
            self.exit2 = NormalTanhPolicy(128, act_dim)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            h0 = self.backbone_norm(self.backbone(obs))
            h1 = self.norm256(self.fc256(h0))
            h2 = self.norm128(self.fc128(h1))
            mean, _ = self.exit2.get_mean_and_std(h2, training=False)
            return torch.tanh(mean)

    class ExitEfullStandalone(nn.Module):
        def __init__(self, obs_dim: int, act_dim: int) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(obs_dim, 512), nn.ReLU(inplace=True))
            self.backbone_norm = UnitRMSNorm(512)
            self.fc256 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True))
            self.norm256 = UnitRMSNorm(256)
            self.fc128 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
            self.norm128 = UnitRMSNorm(128)
            self.trunk_efull = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
            )
            self.norm_efull = UnitRMSNorm(128)
            self.exit3 = NormalTanhPolicy(128, act_dim)

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            h0 = self.backbone_norm(self.backbone(obs))
            h1 = self.norm256(self.fc256(h0))
            h2 = self.norm128(self.fc128(h1))
            h3 = self.norm_efull(self.trunk_efull(h2))
            mean, _ = self.exit3.get_mean_and_std(h3, training=False)
            return torch.tanh(mean)

    return {"e1": Exit1Standalone, "e2": Exit2Standalone, "efull": ExitEfullStandalone}


def load_standalone(
    model_path: Path,
    device: torch.device,
    exit_cls: dict[ExitName, type[nn.Module]],
) -> nn.Module:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    tag: ExitName = ckpt["tag"]
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    net: nn.Module = exit_cls[tag](obs_dim, act_dim)
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device).eval()


def _standalone_paths(out_dir: Path) -> dict[ExitName, Path]:
    return {tag: out_dir / f"{tag}.pt" for tag in ("e1", "e2", "efull")}


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


def _obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


def _timed_forward(
    net: nn.Module, obs: np.ndarray, device: torch.device
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        act = net(_obs_tensor(obs, device)).detach().cpu().numpy()
    if act.ndim == 1:
        act = act.reshape(1, -1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return act, (time.perf_counter() - t0) * 1e3


def _trim_skip(n_steps: int, trim_frac: float = TRIM_FRAC) -> int:
    skip = int(np.ceil(n_steps * trim_frac))
    if skip >= n_steps:
        skip = max(n_steps - 1, 0)
    return skip


def _run_episode_timed(
    net: nn.Module,
    env,
    *,
    scale_action_range: bool,
    action_range: float,
    fixed: torch.Tensor,
    device: torch.device,
    record_latency: bool,
) -> tuple[list[float], float, int]:
    import genesis as gs

    obs_buf, _ = env.reset()
    env.commands.copy_(fixed)
    obs = obs_buf.detach().cpu().numpy()

    latencies: list[float] = []
    reward_sum = 0.0
    step_count = 0
    max_steps = int(env.max_episode_length) + 50

    for _ in range(max_steps):
        act, dt_ms = _timed_forward(net, obs, device)
        step_count += 1
        if record_latency:
            latencies.append(dt_ms)
        if scale_action_range:
            act = act * action_range
        act_t = torch.as_tensor(act, dtype=gs.tc_float, device=gs.device)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(0)
        obs_t, rews, dones, _ = env.step(act_t)
        env.commands.copy_(fixed)
        obs = obs_t.detach().cpu().numpy()
        reward_sum += float(rews[0].item())
        if bool(dones[0].item()):
            n = len(latencies) if record_latency else step_count
            return latencies, reward_sum, n

    raise RuntimeError(f"Exceed {max_steps} step not finished")


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


def _latency_stats(times: np.ndarray) -> dict[str, float]:
    d = {
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "std": float(times.std(ddof=0)),
    }
    for p in _PCTS:
        d[f"p{p}"] = float(np.percentile(times, p))
    return d


def _print_latency(title: str, times: np.ndarray, *, extra: str = "") -> None:
    d = _latency_stats(times)
    suffix = f" | {extra}" if extra else ""
    print(f"\n=== {title}{suffix} ===")
    print(
        f"  n={len(times)}  "
        f"p1={d['p1']:.4f}  p5={d['p5']:.4f}  p10={d['p10']:.4f}  "
        f"mean={d['mean']:.4f}  median={d['median']:.4f}  std={d['std']:.4f}  "
        f"p90={d['p90']:.4f}  p95={d['p95']:.4f}  p97={d['p97']:.4f}  "
        f"p99={d['p99']:.4f}  (ms/step)"
    )


def _eval_exit(
    net: nn.Module,
    tag: ExitName,
    arch: str,
    *,
    backend: BackendName,
    device: torch.device,
    use_cpu: bool,
    n_episodes: int,
) -> dict[str, float]:
    print(f"\n{'=' * 72}")
    print(f"exit [{tag}] {arch}")

    _init_genesis(use_cpu, GS_SEED)
    import genesis as gs

    env = _create_env()
    action_range = float(env.env_cfg["action_range"])
    scale_ar = backend == "standalone"
    fixed = torch.as_tensor([[VX_CMD, VY_CMD, YAW_CMD]], dtype=gs.tc_float, device=gs.device)
    _pin_commands(env, fixed)
    if scale_ar:
        print(f"  action_range={action_range}(FlashSAC: tanh output × scale)")
    else:
        print(
            f"  PPO mean direct step; env clip action_range={action_range} "
            f"action_scale={float(env.env_cfg['action_scale'])}"
        )

    print("  warmup 1 episode (untimed)")
    _run_episode_timed(
        net,
        env,
        scale_action_range=scale_ar,
        action_range=action_range,
        fixed=fixed,
        device=device,
        record_latency=False,
    )

    all_trimmed: list[float] = []
    returns: list[float] = []
    for ep in range(n_episodes):
        lats, ret, n_steps = _run_episode_timed(
            net,
            env,
            scale_action_range=scale_ar,
            action_range=action_range,
            fixed=fixed,
            device=device,
            record_latency=True,
        )
        skip = _trim_skip(len(lats))
        trimmed = lats[skip:]
        all_trimmed.extend(trimmed)
        returns.append(ret)
        print(
            f"  ep {ep + 1}/{n_episodes} return={ret:.3f} steps={n_steps} "
            f"skip={skip} timed={len(trimmed)}"
        )

    _genesis_teardown()

    arr = np.asarray(all_trimmed, dtype=np.float64)
    mean_ret = float(np.mean(returns))
    label = "on-policy PPO" if backend == "onpolicy" else "standalone"
    _print_latency(
        f"{tag} Reasoning takes time ({label}, go before each round {TRIM_FRAC*100:.0f}% and then summarized)",
        arr,
        extra=f"episodes={n_episodes} mean_return={mean_ret:.4f}",
    )
    stats = _latency_stats(arr)
    stats["mean_return"] = mean_ret
    stats["n"] = float(len(arr))
    return stats


def _print_summary_table(rows: list[tuple[ExitName, dict[str, float]]], *, backend: BackendName) -> None:
    label = "on-policy PPO" if backend == "onpolicy" else "standalone"
    print(f"\n{'=' * 72}")
    print(f"Summary of three-person reasoning time consumption ({label}, ms/step, go to the first 1% of each round)")
    print(
        f"{'exit':<8} {'n':>8} {'mean':>8} {'median':>8} {'std':>8} "
        f"{'p90':>8} {'p95':>8} {'p97':>8} {'p99':>8}"
    )
    for tag, d in rows:
        print(
            f"{tag:<8} {int(d['n']):>8} {d['mean']:>8.4f} {d['median']:>8.4f} {d['std']:>8.4f} "
            f"{d['p90']:>8.4f} {d['p95']:>8.4f} {d['p97']:>8.4f} {d['p99']:>8.4f}"
        )


def run_onpolicy(
    model_paths: dict[ExitName, Path],
    *,
    use_cpu: bool = True,
    n_episodes: int = N_EPISODES,
) -> None:
    _setup_cpu()
    device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    missing = [str(p) for p in model_paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("on-policy weight not found:\n" + "\n".join(missing))

    print(f"[time_calculation] backend=onpolicy  device={device}  core={CPU_CORE}")
    for tag in ("e1", "e2", "efull"):
        print(f"  [{tag}] {model_paths[tag]}")

    print(
        f"\nTask: go2_walk_easy vx={VX_CMD:+.3f} vy={VY_CMD:+.3f} yaw={YAW_CMD:+.3f} | "
        f"warmup=1 ep/exit | formal={n_episodes} ep/exit | GO START ={TRIM_FRAC*100:.0f}%"
    )

    summary_rows: list[tuple[ExitName, dict[str, float]]] = []
    try:
        for tag, _, arch in _EXITS:
            net = load_onpolicy_ppo(model_paths[tag], tag, device)
            stats = _eval_exit(
                net,
                tag,
                arch,
                backend="onpolicy",
                device=device,
                use_cpu=use_cpu,
                n_episodes=n_episodes,
            )
            summary_rows.append((tag, stats))
        _print_summary_table(summary_rows, backend="onpolicy")
    finally:
        _genesis_teardown()


def run_standalone(
    standalone_dir: Path,
    *,
    use_cpu: bool = True,
    n_episodes: int = N_EPISODES,
) -> None:
    _setup_cpu()
    device = torch.device("cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    layer_mod = _import_flashsac_layer()
    exit_cls = _build_exit_classes(layer_mod)

    model_paths = _standalone_paths(standalone_dir)
    missing = [str(p) for p in model_paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Standalone weight not found:\n" + "\n".join(missing)
        )

    print(f"[time_calculation] backend=standalone  device={device}  core={CPU_CORE}")
    print(f"Independent model directory: {standalone_dir}")
    for tag, p in model_paths.items():
        print(f"  [{tag}] {p}")

    summary_rows: list[tuple[ExitName, dict[str, float]]] = []
    try:
        for tag in ("e1", "e2", "efull"):
            arch = _STANDALONE_ARCH[tag]
            net = load_standalone(model_paths[tag], device, exit_cls)
            stats = _eval_exit(
                net,
                tag,
                arch,
                backend="standalone",
                device=device,
                use_cpu=use_cpu,
                n_episodes=n_episodes,
            )
            summary_rows.append((tag, stats))
        _print_summary_table(summary_rows, backend="standalone")
    finally:
        _genesis_teardown()


def main() -> None:
    p = argparse.ArgumentParser(description="Go2 three-port reasoning time-consuming distribution")
    p.add_argument(
        "--backend",
        choices=("onpolicy", "standalone"),
        default="onpolicy",
        help="onpolicy=PPO ef/e1/e2 (default); standalone=FlashSAC split .pt",
    )
    p.add_argument("--gpu", action="store_true", help="Use GPU backend instead (default CPU)")
    p.add_argument("--n-episodes", type=int, default=N_EPISODES)
    p.add_argument(
        "--standalone-dir",
        type=Path,
        default=STANDALONE_DIR,
        help="FlashSAC standalone directory",
    )
    p.add_argument("--model-ef", type=Path, default=DEFAULT_ONPOLICY_MODELS["efull"])
    p.add_argument("--model-e1", type=Path, default=DEFAULT_ONPOLICY_MODELS["e1"])
    p.add_argument("--model-e2", type=Path, default=DEFAULT_ONPOLICY_MODELS["e2"])
    args = p.parse_args()

    use_cpu = not args.gpu
    n_episodes = int(args.n_episodes)

    if args.backend == "onpolicy":
        paths: dict[ExitName, Path] = {
            "efull": args.model_ef.expanduser().resolve(),
            "e1": args.model_e1.expanduser().resolve(),
            "e2": args.model_e2.expanduser().resolve(),
        }
        run_onpolicy(paths, use_cpu=use_cpu, n_episodes=n_episodes)
    else:
        run_standalone(args.standalone_dir.expanduser().resolve(), use_cpu=use_cpu, n_episodes=n_episodes)


if __name__ == "__main__":
    main()
