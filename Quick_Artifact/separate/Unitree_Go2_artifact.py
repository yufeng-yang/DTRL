"""Evaluate Unitree Go2 simulation for the main results in Table 2.

Loads PPO and FlashSAC checkpoints. Pins this process to ``CPU_CORE``
(default 0) and runs Genesis on CPU. Does not train.

Classes: ExitActor, WeightedRMSNorm, MeanOnlyOffActor, LatencyBuffer,
EpisodeStats.
Functions: _setup_cpu, _genesis_teardown, _command_tensor, _obs_tensor,
_velocity_errors, _episode_success, _timed_infer, _load_ppo,
_build_on_exits, _build_mms_exits, _load_optimized_off, _build_off_exits,
_select_exit, _run_env_steps, _warmup_exit_into_buffer, _run_episode,
_summarize, _print_progress, _eval_method, main.
"""

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE = Path(__file__).resolve().parents[2]
_ROOT = _CODE / "Src_Training" / "Unitree Go2 Sim"
_EVAL = _ROOT / "evaluation"
_FLASHSAC = _CODE / "extra_resources" / "FlashSAC"
if not _FLASHSAC.is_dir():
    _FLASHSAC = _CODE / "repro" / "extra_resources" / "FlashSAC"

if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from time_calculation import (
    PpoE1Actor,
    PpoE2Actor,
    PpoEfullActor,
    _create_env,
    _pin_commands,
    load_onpolicy_ppo,
    _standalone_paths,
)

TASK_NAME = "Unitree Go2 Sim"
# Tunable evaluation knobs (each name is commented; see README.md).
CPU_CORE = 0                 # CPU to pin; falls back if this core is missing
GS_SEED = 0                  # Genesis scene seed
WARMUP_STEPS = 1000          # env steps used to fill the latency buffer
PROFILE_STEPS = 2000         # extra profiling steps written into the buffer
N_EPISODES = 100             # number of evaluation episodes
START_SEED = 42              # first episode seed (then +1, +2, …)
DECISION_INTERVAL = 10       # control steps between exit re-selections
BUFFER_SIZE = 2000           # rolling window of measured inference latencies

P_EXIT_ESTIMATE = 95         # percentile used as an exit's predicted latency (ms)
BUDGET_LOW_TAG = "e1"        # exit that sets the lower end of the deadline band
P_BUDGET_LOW = 92            # percentile of that exit's measured latency
BUDGET_HIGH_TAG = "ef"       # exit that sets the upper end of the deadline band
P_BUDGET_HIGH = 98           # percentile of that exit's measured latency

VY_CMD = 0.0                 # commanded lateral velocity
YAW_CMD = 0.0                # commanded yaw rate
EV_THRESHOLD = 0.1           # forward-speed error below this counts as tracking
STARTUP_SKIP_STEPS = 50      # leading steps ignored when scoring tracking
_V_CMD_EPS = 1e-8            # avoid divide-by-zero in velocity error

# Each entry corresponds to one available computation level.
EXIT_DEPTH = {"e1": 1, "e2": 2, "ef": 3}

FULL_PT = _ROOT / "Full/full_policy/Full_20260516_092741_449462/Full_8000.pt"
SWI_PT = _ROOT / "SWI/runs/SWI_20260516_090620_581385/SWI_999.pt"
MMS_E2_PT = _ROOT / "MMS/runs/MMS_20260516_105335_797134/MMS_3500.pt"
ON_E1 = _ROOT / "DTRL-On/runs/onpolicy/DTRL-On_e1_20260516_102657_632627/DTRL-On_e1_2999.pt"
ON_E2 = _ROOT / "DTRL-On/runs/onpolicy/DTRL-On_e2_20260516_104308_450115/DTRL-On_e2_2999.pt"
ON_EF = _ROOT / "DTRL-On/runs/onpolicy/DTRL-On_full_20260516_092741_449462/DTRL-On_full_8000.pt"
STANDALONE_DIR = _EVAL / "standalone_models" / "step48825"


@dataclass
class ExitActor:
    tag: str
    actor: nn.Module
    depth: int
    scale_action: bool  # FlashSAC standalone needs action_range


class WeightedRMSNorm(nn.Module):
    """Inference-only equivalent of FlashSAC UnitRMSNorm."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, eps=1e-6)


class MeanOnlyOffActor(nn.Module):
    """DTRL-OFF actor without the unused stochastic standard-deviation head."""

    def __init__(self, tag: str, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.tag = tag
        self.backbone = nn.Linear(obs_dim, 512)
        self.backbone_norm = WeightedRMSNorm(512)
        if tag != "e1":
            self.fc256 = nn.Linear(512, 256)
            self.norm256 = WeightedRMSNorm(256)
            self.fc128 = nn.Linear(256, 128)
            self.norm128 = WeightedRMSNorm(128)
        if tag == "efull":
            self.fc128_a = nn.Linear(128, 128)
            self.fc128_b = nn.Linear(128, 128)
            self.norm_efull = WeightedRMSNorm(128)
        mean_in = 512 if tag == "e1" else 128
        self.mean = nn.Linear(mean_in, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone_norm(F.relu(self.backbone(obs)))
        if self.tag != "e1":
            h = self.norm256(F.relu(self.fc256(h)))
            h = self.norm128(F.relu(self.fc128(h)))
        if self.tag == "efull":
            h = F.relu(self.fc128_a(h))
            h = F.relu(self.fc128_b(h))
            h = self.norm_efull(h)
        return torch.tanh(self.mean(h))


class LatencyBuffer:
    def __init__(self, capacity: int = BUFFER_SIZE) -> None:
        self._buf: deque[float] = deque(maxlen=capacity)

    def push(self, latency_ms: float) -> None:
        self._buf.append(latency_ms)

    def percentile_ms(self, p: float) -> float:
        if not self._buf:
            return float("inf")
        return float(np.percentile(np.asarray(self._buf, dtype=np.float64), p))

    def p_estimate_ms(self) -> float:
        return self.percentile_ms(P_EXIT_ESTIMATE)

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class EpisodeStats:
    return_: float
    n_steps: int
    deadline_hits: int
    success: bool
    e_x: float
    survived: bool


def _setup_cpu() -> None:
    if hasattr(os, "sched_setaffinity"):
        available = sorted(os.sched_getaffinity(0))
        core = CPU_CORE if CPU_CORE in available else available[0]
        os.sched_setaffinity(0, {core})
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


def _command_tensor(vx: float) -> torch.Tensor:
    import genesis as gs

    return torch.tensor([[vx, VY_CMD, YAW_CMD]], dtype=gs.tc_float, device=gs.device)


def _obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


def _velocity_errors(
    vels: list[tuple[float, float]], *, v_cmd: tuple[float, float]
) -> tuple[float, float, float]:
    if not vels:
        return 0.0, 0.0, 0.0
    arr = np.asarray(vels, dtype=np.float64)
    vx_cmd, vy_cmd = v_cmd
    denom = max(float(np.hypot(vx_cmd, vy_cmd)), _V_CMD_EPS)
    e_x = float(np.mean(np.abs(arr[:, 0] - vx_cmd)) / denom)
    e_y = float(np.mean(np.abs(arr[:, 1] - vy_cmd)) / denom)
    e_v = float(
        np.mean(np.hypot(arr[:, 0] - vx_cmd, arr[:, 1] - vy_cmd)) / denom
    )
    return e_v, e_x, e_y


def _episode_success(*, survived: bool, e_x: float) -> bool:
    return bool(survived and e_x < EV_THRESHOLD)


def _timed_infer(
    exit_actor: ExitActor,
    obs: np.ndarray,
    device: torch.device,
    action_range: float,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        act = exit_actor.actor(_obs_tensor(obs, device)).detach().cpu().numpy()
    if act.ndim == 1:
        act = act.reshape(1, -1)
    if exit_actor.scale_action:
        act = act * action_range
    if device.type == "cuda":
        torch.cuda.synchronize()
    return act, (time.perf_counter() - t0) * 1e3


def _load_ppo(path: Path, tag: str, device: torch.device) -> nn.Module:
    # map ef -> efull for loader
    loader_tag = "efull" if tag == "ef" else tag
    return load_onpolicy_ppo(path, loader_tag, device)


def _build_on_exits(device: torch.device) -> dict[str, ExitActor]:
    paths = {"e1": ON_E1, "e2": ON_E2, "ef": ON_EF}
    out: dict[str, ExitActor] = {}
    for tag, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        net = _load_ppo(path, tag, device)
        out[tag] = ExitActor(tag=tag, actor=net, depth=EXIT_DEPTH[tag], scale_action=False)
    return out


def _build_mms_exits(device: torch.device) -> dict[str, ExitActor]:
    specs: list[tuple[str, Path]] = [
        ("e1", SWI_PT),
        ("e2", MMS_E2_PT),
        ("ef", FULL_PT),
    ]
    out: dict[str, ExitActor] = {}
    for tag, path in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        net = _load_ppo(path, tag, device)
        out[tag] = ExitActor(tag=tag, actor=net, depth=EXIT_DEPTH[tag], scale_action=False)
    return out


def _load_optimized_off(path: Path, src_tag: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("tag") != src_tag:
        raise ValueError(
            f"{path}: expected tag={src_tag!r}, got {checkpoint.get('tag')!r}"
        )
    state = checkpoint["state_dict"]
    net = MeanOnlyOffActor(
        src_tag,
        int(checkpoint["obs_dim"]),
        int(checkpoint["act_dim"]),
    )
    mapped = {
        "backbone.weight": state["backbone.0.weight"],
        "backbone.bias": state["backbone.0.bias"],
        "backbone_norm.weight": state["backbone_norm.weight"],
    }
    if src_tag != "e1":
        mapped.update(
            {
                "fc256.weight": state["fc256.0.weight"],
                "fc256.bias": state["fc256.0.bias"],
                "norm256.weight": state["norm256.weight"],
                "fc128.weight": state["fc128.0.weight"],
                "fc128.bias": state["fc128.0.bias"],
                "norm128.weight": state["norm128.weight"],
            }
        )
    if src_tag == "efull":
        mapped.update(
            {
                "fc128_a.weight": state["trunk_efull.0.weight"],
                "fc128_a.bias": state["trunk_efull.0.bias"],
                "fc128_b.weight": state["trunk_efull.2.weight"],
                "fc128_b.bias": state["trunk_efull.2.bias"],
                "norm_efull.weight": state["norm_efull.weight"],
            }
        )
    head = "exit3" if src_tag == "efull" else src_tag.replace("e", "exit")
    mapped["mean.weight"] = state[f"{head}.mean_w.w.weight"]
    mapped["mean.bias"] = state[f"{head}.mean_bias"]
    net.load_state_dict(mapped, strict=True)
    net.eval()

    example = torch.zeros(
        (1, int(checkpoint["obs_dim"])),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        traced = torch.jit.trace(net, example, check_trace=True)
        return torch.jit.freeze(traced.eval()).to(device)


def _build_off_exits(device: torch.device) -> dict[str, ExitActor]:
    out: dict[str, ExitActor] = {}
    mapping: list[tuple[str, str]] = [("e1", "e1"), ("e2", "e2"), ("ef", "efull")]
    for tag, src_tag in mapping:
        p = STANDALONE_DIR / f"{src_tag}.pt"
        if not p.is_file():
            raise FileNotFoundError(p)
        net = _load_optimized_off(p, src_tag, device)
        out[tag] = ExitActor(tag=tag, actor=net, depth=EXIT_DEPTH[tag], scale_action=True)
    return out


def _select_exit(
    deadline_ms: float,
    buffers: dict[str, LatencyBuffer],
    exits: dict[str, ExitActor],
) -> str:
    ordered = sorted(exits.values(), key=lambda x: x.depth, reverse=True)
    for ex in ordered:
        if buffers[ex.tag].p_estimate_ms() <= deadline_ms:
            return ex.tag
    return ordered[-1].tag


def _run_env_steps(
    env,
    exit_actor: ExitActor,
    buf: LatencyBuffer | None,
    *,
    device: torch.device,
    action_range: float,
    n_steps: int,
    record_latency: bool,
    vx: float,
) -> None:
    import genesis as gs

    cmd = _command_tensor(vx)
    _pin_commands(env, cmd)
    obs_buf, _ = env.reset()
    env.commands.copy_(cmd)
    obs = obs_buf.detach().cpu().numpy()
    done = False
    n = 0
    while n < n_steps:
        if done:
            cmd = _command_tensor(vx)
            _pin_commands(env, cmd)
            obs_buf, _ = env.reset()
            env.commands.copy_(cmd)
            obs = obs_buf.detach().cpu().numpy()
            done = False
        act, lat_ms = _timed_infer(exit_actor, obs, device, action_range)
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


def _warmup_exit_into_buffer(
    env,
    exit_actor: ExitActor,
    buf: LatencyBuffer,
    tag: str,
    *,
    device: torch.device,
    action_range: float,
    vx: float,
) -> None:
    print(f"[profiling] {tag}", flush=True)
    _run_env_steps(
        env,
        exit_actor,
        None,
        device=device,
        action_range=action_range,
        n_steps=WARMUP_STEPS,
        record_latency=False,
        vx=vx,
    )
    _run_env_steps(
        env,
        exit_actor,
        buf,
        device=device,
        action_range=action_range,
        n_steps=PROFILE_STEPS,
        record_latency=True,
        vx=vx,
    )


def _run_episode(
    env,
    exits: dict[str, ExitActor],
    buffers: dict[str, LatencyBuffer],
    *,
    device: torch.device,
    action_range: float,
    rng: np.random.Generator,
    budget_low_ms: float,
    budget_high_ms: float,
    fixed_tag: str | None,
    vx: float,
) -> EpisodeStats:
    import genesis as gs

    cmd = _command_tensor(vx)
    _pin_commands(env, cmd)
    obs_buf, _ = env.reset()
    env.commands.copy_(cmd)
    obs = obs_buf.detach().cpu().numpy()

    ep_ret = 0.0
    step = 0
    hits = 0
    vels: list[tuple[float, float]] = []
    deadline_ms = budget_high_ms
    active: str = "e1" if fixed_tag is None else fixed_tag
    last_action: np.ndarray | None = None
    done = False
    survived = False
    max_steps = int(env.max_episode_length) + 50

    while not done and step < max_steps:
        if step % DECISION_INTERVAL == 0:
            deadline_ms = float(rng.uniform(budget_low_ms, budget_high_ms))
            if fixed_tag is None:
                active = _select_exit(deadline_ms, buffers, exits)

        ex = exits[active]
        act, lat_ms = _timed_infer(ex, obs, device, action_range)
        buffers[active].push(lat_ms)

        if lat_ms <= deadline_ms:
            action = act
            last_action = action
            hit = True
        elif last_action is not None:
            action = last_action
            hit = False
        else:
            action = act
            last_action = action
            hit = True

        act_t = torch.as_tensor(action, dtype=gs.tc_float, device=gs.device)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(0)
        obs_t, rews, dones, extras = env.step(act_t)
        env.commands.copy_(cmd)
        obs = obs_t.detach().cpu().numpy()
        ep_ret += float(rews[0].item())
        step += 1
        if hit:
            hits += 1
        if step > STARTUP_SKIP_STEPS:
            vels.append(
                (
                    float(env.base_lin_vel[0, 0].item()),
                    float(env.base_lin_vel[0, 1].item()),
                )
            )
        done = bool(dones[0].item())
        if done:
            time_outs = extras.get("time_outs")
            if time_outs is not None:
                survived = float(time_outs[0].item()) > 0.5
            else:
                survived = step >= int(env.max_episode_length)

    _, e_x, _ = _velocity_errors(vels, v_cmd=(vx, VY_CMD))
    return EpisodeStats(
        return_=ep_ret,
        n_steps=step,
        deadline_hits=hits,
        success=_episode_success(survived=survived, e_x=e_x),
        e_x=e_x,
        survived=survived,
    )


def _summarize(
    name: str, results: list[EpisodeStats], budget_low: float, budget_high: float
) -> None:
    rets = np.array([r.return_ for r in results], dtype=np.float64)
    hit_rates = np.array(
        [r.deadline_hits / max(r.n_steps, 1) for r in results], dtype=np.float64
    )
    success_rate = float(np.mean([r.success for r in results]))
    print(f"\n{'=' * 70}")
    print(f"summary ({N_EPISODES} episodes) — {name}:")
    print(f"  budget (ms)     = [{budget_low:.4f}, {budget_high:.4f}]")
    print(f"  mean return     = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  mean hit rate   = {hit_rates.mean():.4f} ± {hit_rates.std(ddof=0):.4f}")
    print(
        f"  success rate    = {success_rate:.4f} "
        f"({sum(r.success for r in results)}/{N_EPISODES})"
    )


def _print_progress(method: str, done: int, total: int) -> None:
    print(
        f"\r[{TASK_NAME}] running {method}: {done}/{total} episodes",
        end="",
        flush=True,
    )
    if done >= total:
        print()


def _eval_method(
    name: str,
    env,
    exits: dict[str, ExitActor],
    buffers: dict[str, LatencyBuffer],
    *,
    device: torch.device,
    action_range: float,
    rng: np.random.Generator,
    budget_low: float,
    budget_high: float,
    fixed_tag: str | None,
    vx: float,
) -> list[EpisodeStats]:
    results: list[EpisodeStats] = []
    _print_progress(name, 0, N_EPISODES)
    for i in range(N_EPISODES):
        results.append(
            _run_episode(
                env,
                exits,
                buffers,
                device=device,
                action_range=action_range,
                rng=rng,
                budget_low_ms=budget_low,
                budget_high_ms=budget_high,
                fixed_tag=fixed_tag,
                vx=vx,
            )
        )
        _print_progress(name, i + 1, N_EPISODES)
    _summarize(name, results, budget_low, budget_high)
    return results


def main() -> None:
    for p in (FULL_PT, SWI_PT, MMS_E2_PT, ON_E1, ON_E2, ON_EF):
        if not p.is_file():
            raise SystemExit(f"missing weight: {p}")
    for p in _standalone_paths(STANDALONE_DIR).values():
        if not p.is_file():
            raise SystemExit(f"missing standalone: {p}")
    if not _FLASHSAC.is_dir():
        raise SystemExit(f"missing FlashSAC: {_FLASHSAC}")

    _setup_cpu()
    device = torch.device("cpu")

    import genesis as gs

    gs.init(backend=gs.cpu, precision="32", seed=GS_SEED, logging_level="warning")
    try:
        env = _create_env()
        action_range = float(env.env_cfg["action_range"])
        rng = np.random.default_rng(START_SEED)

        on_exits = _build_on_exits(device)
        on_buffers: dict[str, LatencyBuffer] = {
            tag: LatencyBuffer() for tag in EXIT_DEPTH
        }
        mms_exits = _build_mms_exits(device)
        mms_buffers: dict[str, LatencyBuffer] = {
            tag: LatencyBuffer() for tag in EXIT_DEPTH
        }
        off_exits = _build_off_exits(device)
        off_buffers: dict[str, LatencyBuffer] = {
            tag: LatencyBuffer() for tag in EXIT_DEPTH
        }
        full_exits = dict(mms_exits)
        full_buffers = {tag: LatencyBuffer() for tag in EXIT_DEPTH}
        swi_exits = dict(mms_exits)
        swi_buffers = {tag: LatencyBuffer() for tag in EXIT_DEPTH}

        low_tag = BUDGET_LOW_TAG
        high_tag = BUDGET_HIGH_TAG
        for tag in ("e1", "e2", "ef"):
            _warmup_exit_into_buffer(
                env,
                on_exits[tag],
                on_buffers[tag],
                f"DTRL-ON {tag}",
                device=device,
                action_range=action_range,
                vx=0.6,
            )
        budget_low = on_buffers[low_tag].percentile_ms(P_BUDGET_LOW)
        budget_high = on_buffers[high_tag].percentile_ms(P_BUDGET_HIGH)
        if budget_low > budget_high:
            budget_low, budget_high = budget_high, budget_low
        for tag in ("e1", "e2", "ef"):
            _warmup_exit_into_buffer(
                env,
                mms_exits[tag],
                mms_buffers[tag],
                f"MMS {tag}",
                device=device,
                action_range=action_range,
                vx=0.6,
            )
        for tag in ("e1", "e2", "ef"):
            _warmup_exit_into_buffer(
                env,
                off_exits[tag],
                off_buffers[tag],
                f"DTRL-OFF {tag}",
                device=device,
                action_range=action_range,
                vx=0.5,
            )
        off_budget_low = off_buffers[low_tag].percentile_ms(P_BUDGET_LOW)
        off_budget_high = off_buffers[high_tag].percentile_ms(P_BUDGET_HIGH)
        if off_budget_low > off_budget_high:
            off_budget_low, off_budget_high = off_budget_high, off_budget_low

        full_results = _eval_method(
            "FULL",
            env,
            full_exits,
            full_buffers,
            device=device,
            action_range=action_range,
            rng=rng,
            budget_low=budget_low,
            budget_high=budget_high,
            fixed_tag="ef",
            vx=0.6,
        )
        swi_results = _eval_method(
            "SWI",
            env,
            swi_exits,
            swi_buffers,
            device=device,
            action_range=action_range,
            rng=rng,
            budget_low=budget_low,
            budget_high=budget_high,
            fixed_tag="e1",
            vx=0.6,
        )
        mms_results = _eval_method(
            "MMS",
            env,
            mms_exits,
            mms_buffers,
            device=device,
            action_range=action_range,
            rng=rng,
            budget_low=budget_low,
            budget_high=budget_high,
            fixed_tag=None,
            vx=0.6,
        )
        on_results = _eval_method(
            "DTRL-ON",
            env,
            on_exits,
            on_buffers,
            device=device,
            action_range=action_range,
            rng=rng,
            budget_low=budget_low,
            budget_high=budget_high,
            fixed_tag=None,
            vx=0.6,
        )
        off_results = _eval_method(
            "DTRL-OFF",
            env,
            off_exits,
            off_buffers,
            device=device,
            action_range=action_range,
            rng=rng,
            budget_low=off_budget_low,
            budget_high=off_budget_high,
            fixed_tag=None,
            vx=0.5,
        )

        print(f"\n{'=' * 70}")
        for name, results in (
            ("FULL", full_results),
            ("SWI", swi_results),
            ("MMS", mms_results),
            ("DTRL-ON", on_results),
            ("DTRL-OFF", off_results),
        ):
            rets = np.array([r.return_ for r in results], dtype=np.float64)
            hits = np.array(
                [r.deadline_hits / max(r.n_steps, 1) for r in results],
                dtype=np.float64,
            )
            succ = float(np.mean([r.success for r in results]))
            print(
                f"  {name:8s}  return={rets.mean():8.4f}±{rets.std(ddof=0):.4f}  "
                f"hit={hits.mean():.4f}±{hits.std(ddof=0):.4f}  "
                f"success={succ:.4f}"
            )
    finally:
        _genesis_teardown()


if __name__ == "__main__":
    main()
