"""Go2 b2_owf SWI shallow PPO fixed strategy formal evaluation (same tasks and metrics as full_eval).

Load b2_owf SWI model_<iter>.pt (obs→512→out), evaluate on go2_walk_easy.
Each round randomly samples vx∈[-1, 1], vy=0, yaw=0.
The PPO output is Gaussian mean (without tanh), which is fed directly into env.step (without multiplication by action_range).

Sampling deadline every 10 steps; miss → reuse the previous step action.
Warmup 1000 steps + fill latency buffer 1000 steps before formal evaluation.
E_v still removes the first 50 steps of each game; deadline hit rate counts all steps in the entire process (including the start).
success = running full episode and E_v < 0.1.

Run::

    python unitree_go2/evaluation/swi_eval.py
    python unitree_go2/evaluation/swi_eval.py --n-episodes 10"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_EVAL = Path(__file__).resolve().parent
_ROOT = _EVAL.parent
_FLASHSAC = _ROOT.parents[1] / "extra_resources" / "FlashSAC"

DEFAULT_MODEL = (
    _ROOT / "SWI/runs/SWI_20260516_090620_581385/SWI_999.pt"
)

VX_CMD_LOW = -1.0
VX_CMD_HIGH = 1.0
VY_CMD = 0.0
YAW_CMD = 0.0

CPU_CORE = 1
GS_SEED = 0
WARMUP_STEPS = 1000
BUFFER_SIZE = 1000
P_ESTIMATE = 97
N_EPISODES = 100
START_SEED = 42
DECISION_INTERVAL = 10
DEADLINE_LOW_MS = 0.09
DEADLINE_HIGH_MS = 0.16
EV_THRESHOLD = 0.1
STARTUP_SKIP_STEPS = 50
_V_CMD_EPS = 1e-8


class LatencyBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self._buf: deque[float] = deque(maxlen=capacity)

    def push(self, latency_ms: float) -> None:
        self._buf.append(latency_ms)

    def p_estimate_ms(self) -> float:
        if not self._buf:
            return float("inf")
        return float(np.percentile(np.asarray(self._buf, dtype=np.float64), P_ESTIMATE))

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class EpisodeStats:
    ep: int
    vx_cmd: float
    return_: float
    n_steps: int
    stat_steps: int
    deadline_hits: int
    deadline_hit_rate: float
    success: bool
    survived: bool
    e_v: float
    e_x: float
    e_y: float


class PpoSwiActor(nn.Module):
    """RSL-RL MLPModel SWI actor: obs→512→act (ELU, deterministic mean)."""

    def __init__(self, obs_dim: int = 45, act_dim: int = 12) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


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


def _load_swi_actor(model_path: Path, device: torch.device) -> PpoSwiActor:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "actor_state_dict" not in ckpt:
        raise TypeError(f"Unable to parse checkpoint: {model_path}")
    raw = ckpt["actor_state_dict"]
    mlp_sd = {k: v for k, v in raw.items() if k.startswith("mlp.")}
    if not mlp_sd:
        raise KeyError(f"{model_path} Missing actor mlp weights")

    obs_dim, act_dim = 45, 12
    w0 = mlp_sd.get("mlp.0.weight")
    w_out = mlp_sd.get("mlp.2.weight")
    if w0 is not None:
        obs_dim = int(w0.shape[1])
    if w_out is not None:
        act_dim = int(w_out.shape[0])

    actor = PpoSwiActor(obs_dim, act_dim).to(device)
    actor.load_state_dict(mlp_sd, strict=True)
    return actor.eval()


def _obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


@torch.inference_mode()
def _policy_action(actor: PpoSwiActor, obs: np.ndarray, device: torch.device) -> np.ndarray:
    act = actor(_obs_tensor(obs, device)).detach().cpu().numpy()
    if act.ndim == 1:
        act = act.reshape(1, -1)
    return act


def _timed_policy_action(
    actor: PpoSwiActor,
    obs: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    act = _policy_action(actor, obs, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return act, (time.perf_counter() - t0) * 1e3


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


def _pin_commands(env, cmd: torch.Tensor) -> None:
    def _pinned(envs_idx=None) -> None:
        if envs_idx is None:
            env.commands.copy_(cmd)
        else:
            idx = envs_idx.nonzero(as_tuple=False).flatten()
            if idx.numel() > 0:
                env.commands[idx] = cmd[idx]

    env._resample_commands = _pinned  # type: ignore[method-assign]


def _sample_vx_cmd(rng: np.random.Generator) -> float:
    return float(rng.uniform(VX_CMD_LOW, VX_CMD_HIGH))


def _command_tensor(vx: float) -> torch.Tensor:
    import genesis as gs

    return torch.tensor([[vx, VY_CMD, YAW_CMD]], dtype=gs.tc_float, device=gs.device)


def _velocity_errors(
    vels: list[tuple[float, float]],
    v_cmd: tuple[float, float],
) -> tuple[float, float, float]:
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


def _episode_success(*, survived: bool, e_v: float) -> bool:
    return survived and e_v < EV_THRESHOLD


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


def _sample_deadline_ms(rng: np.random.Generator) -> float:
    return float(rng.uniform(DEADLINE_LOW_MS, DEADLINE_HIGH_MS))


def _run_env_steps(
    env,
    actor: PpoSwiActor,
    buf: LatencyBuffer | None,
    *,
    rng: np.random.Generator,
    device: torch.device,
    n_steps: int,
    record_latency: bool,
) -> None:
    import genesis as gs

    vx = _sample_vx_cmd(rng)
    cmd = _command_tensor(vx)
    _pin_commands(env, cmd)

    obs_buf, _ = env.reset()
    env.commands.copy_(cmd)
    obs = obs_buf.detach().cpu().numpy()
    done = False
    n = 0
    while n < n_steps:
        if done:
            vx = _sample_vx_cmd(rng)
            cmd = _command_tensor(vx)
            _pin_commands(env, cmd)
            obs_buf, _ = env.reset()
            env.commands.copy_(cmd)
            obs = obs_buf.detach().cpu().numpy()
            done = False

        act, lat_ms = _timed_policy_action(actor, obs, device)
        if record_latency and buf is not None:
            buf.push(lat_ms)

        act_t = torch.as_tensor(act, dtype=gs.tc_float, device=gs.device)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(0)
        obs_t, _, dones, _ = env.step(act_t)
        env.commands.copy_(cmd)
        obs = obs_t.detach().cpu().numpy()
        done = bool(dones[0].item())
        n += 1


def _warmup(
    actor: PpoSwiActor,
    env,
    *,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    _run_env_steps(
        env,
        actor,
        None,
        rng=rng,
        device=device,
        n_steps=WARMUP_STEPS,
        record_latency=False,
    )
    print(f"  warmup {WARMUP_STEPS} step (do not write buffer, core={CPU_CORE})")


def _fill_buffer(
    actor: PpoSwiActor,
    env,
    buf: LatencyBuffer,
    *,
    rng: np.random.Generator,
    device: torch.device,
) -> None:
    _run_env_steps(
        env,
        actor,
        buf,
        rng=rng,
        device=device,
        n_steps=BUFFER_SIZE,
        record_latency=True,
    )
    print(
        f"  Fill buffer {BUFFER_SIZE} step | len={len(buf)} | "
        f"p{P_ESTIMATE}={buf.p_estimate_ms():.4f} ms"
    )


def _run_episode(
    env,
    actor: PpoSwiActor,
    buf: LatencyBuffer,
    *,
    device: torch.device,
    ep_idx: int,
    rng: np.random.Generator,
) -> EpisodeStats:
    import genesis as gs

    vx_cmd = _sample_vx_cmd(rng)
    cmd = _command_tensor(vx_cmd)
    _pin_commands(env, cmd)

    obs_buf, _ = env.reset()
    env.commands.copy_(cmd)
    obs = obs_buf.detach().cpu().numpy()

    ep_ret = 0.0
    step = 0
    hits = 0
    stat_steps = 0
    vels: list[tuple[float, float]] = []
    deadline_ms = DEADLINE_HIGH_MS
    last_action: np.ndarray | None = None
    done = False
    survived = False

    max_steps = int(env.max_episode_length) + 50
    while not done and step < max_steps:
        if step % DECISION_INTERVAL == 0:
            deadline_ms = _sample_deadline_ms(rng)

        act, lat_ms = _timed_policy_action(actor, obs, device)
        buf.push(lat_ms)

        if lat_ms <= deadline_ms:
            action = act
            last_action = action
            hit_this_step = True
        elif last_action is not None:
            action = last_action
            hit_this_step = False
        else:
            action = act
            last_action = action
            hit_this_step = True

        act_t = torch.as_tensor(action, dtype=gs.tc_float, device=gs.device)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(0)
        obs_t, rews, dones, extras = env.step(act_t)
        env.commands.copy_(cmd)
        obs = obs_t.detach().cpu().numpy()
        ep_ret += float(rews[0].item())
        step += 1
        if hit_this_step:
            hits += 1
        if step > STARTUP_SKIP_STEPS:
            stat_steps += 1
            vels.append(
                (float(env.base_lin_vel[0, 0].item()), float(env.base_lin_vel[0, 1].item()))
            )
        done = bool(dones[0].item())

        if done:
            time_outs = extras.get("time_outs")
            if time_outs is not None:
                survived = float(time_outs[0].item()) > 0.5
            else:
                survived = step >= int(env.max_episode_length)

    e_v, e_x, e_y = _velocity_errors(vels, v_cmd=(vx_cmd, VY_CMD))
    hit_rate = hits / max(step, 1)
    return EpisodeStats(
        ep=ep_idx,
        vx_cmd=vx_cmd,
        return_=ep_ret,
        n_steps=step,
        stat_steps=stat_steps,
        deadline_hits=hits,
        deadline_hit_rate=hit_rate,
        success=_episode_success(survived=survived, e_v=e_v),
        survived=survived,
        e_v=e_v,
        e_x=e_x,
        e_y=e_y,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 b2_owf SWI shallow PPO formal evaluation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (default CPU)")
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--rng-seed", type=int, default=START_SEED)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="b2_owf SWI model_<iter>.pt",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Weight not found: {model_path}")

    use_cpu = not args.gpu
    device = torch.device("cpu" if use_cpu else "cuda")
    n_episodes = int(args.n_episodes)
    rng = np.random.default_rng(int(args.rng_seed))

    _setup_cpu()
    actor = _load_swi_actor(model_path, device)

    print(f"Weight: {model_path}")
    print("Strategy: b2_owf SWI shallow PPO (obs→512→out)")
    print(
        f"Task: go2_walk_easy vx∈[{VX_CMD_LOW},{VX_CMD_HIGH}] Uniformly random in each round, "
        f"vy={VY_CMD}, yaw={YAW_CMD}"
    )
    print(f"equipment: {device} | Single env reuse | core={CPU_CORE}")
    print(
        f"deadline∈[{DEADLINE_LOW_MS},{DEADLINE_HIGH_MS}] ms | every{DECISION_INTERVAL} step resampling | "
        f"buffer={BUFFER_SIZE} p{P_ESTIMATE} | episodes={n_episodes} | rng_seed={args.rng_seed}"
    )
    print(
        f"success: run full episode (timeout) and start {STARTUP_SKIP_STEPS} after step "
        f"E_v < {EV_THRESHOLD} (E_x/E_y synthesis, ||v_cmd|| normalization)"
    )
    print("Action: PPO mean direct step (not multiplied by action_range)")
    print(
        f"\nPreheat + fill buffer (single core {CPU_CORE}):\n"
        f"  1) warmup {WARMUP_STEPS} step\n"
        f"  2) Run again {BUFFER_SIZE} Fill buffer step by step"
    )

    _init_genesis(use_cpu, GS_SEED)

    env = _create_env()
    action_range = float(env.env_cfg["action_range"])
    print(
        f"env action_range={action_range}(clip only) "
        f"action_scale={float(env.env_cfg['action_scale'])}"
    )

    latency_buf = LatencyBuffer()
    _warmup(actor, env, rng=rng, device=device)
    _fill_buffer(actor, env, latency_buf, rng=rng, device=device)

    results: list[EpisodeStats] = []
    print("\nFormal evaluation:")
    try:
        for ep in range(n_episodes):
            st = _run_episode(
                env,
                actor,
                latency_buf,
                device=device,
                ep_idx=ep,
                rng=rng,
            )
            results.append(st)
            print(
                f"  ep {ep + 1}/{n_episodes} vx={st.vx_cmd:.4f} return={st.return_:.4f} "
                f"steps={st.n_steps} survived={st.survived} "
                f"E_v={st.e_v:.4f} E_x={st.e_x:.4f} E_y={st.e_y:.4f} "
                f"deadline_hit_rate={st.deadline_hit_rate:.4f} "
                f"({st.deadline_hits}/{st.n_steps}) success={st.success}"
            )
            print(f"    buffer p{P_ESTIMATE}={latency_buf.p_estimate_ms():.4f} ms")
    finally:
        _genesis_teardown()

    rets = np.array([r.return_ for r in results], dtype=np.float64)
    steps = np.array([r.n_steps for r in results], dtype=np.float64)
    evs = np.array([r.e_v for r in results], dtype=np.float64)
    exs = np.array([r.e_x for r in results], dtype=np.float64)
    eys = np.array([r.e_y for r in results], dtype=np.float64)
    hit_rates = np.array([r.deadline_hit_rate for r in results], dtype=np.float64)
    total_deadline_hits = int(sum(r.deadline_hits for r in results))
    total_steps = int(sum(r.n_steps for r in results))
    deadline_hit_rate = total_deadline_hits / max(total_steps, 1)
    success_rate = float(np.mean([r.success for r in results]))
    survive_rate = float(np.mean([r.survived for r in results]))

    print(f"\n{'=' * 70}")
    print(f"comprehensive ({n_episodes} episodes) — Go2 b2_owf SWI PPO:")
    print(f"  mean return     = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  average step    = {steps.mean():.4f} ± {steps.std(ddof=0):.4f}")
    print(f"  mean E_v        = {evs.mean():.4f} ± {evs.std(ddof=0):.4f}")
    print(f"  mean E_x        = {exs.mean():.4f} ± {exs.std(ddof=0):.4f}")
    print(f"  mean E_y        = {eys.mean():.4f} ± {eys.std(ddof=0):.4f}")
    print(
        f"  mean hit rate   = {hit_rates.mean():.4f} ± {hit_rates.std(ddof=0):.4f} "
        f"(per-episode, full step)"
    )
    print(
        f"  deadline hit rate = {deadline_hit_rate:.4f} "
        f"({total_deadline_hits}/{total_steps}, the whole process step by step, including the start)"
    )
    print(f"  survive rate    = {survive_rate:.4f}")
    print(f"  success rate    = {success_rate:.4f} ({sum(r.success for r in results)}/{n_episodes})")
    print(f"  buffer p{P_ESTIMATE} (ms) = {latency_buf.p_estimate_ms():.4f}")


if __name__ == "__main__":
    main()
