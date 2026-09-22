"""Evaluate Semicircle-Narrow for Table 2 and the Figure 8 comparison.

Environment ``SafetyPointSemicircle0-v5``. Loads trained weights only.
Pins this process to ``CPU_CORE`` (default 0). Same class/function layout
as ``Semicircle_Wide_artifact.py``.
"""

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO, SAC

_CODE = Path(__file__).resolve().parents[2]
_ROOT = _CODE / "Src_Training" / "Semicircle-Narrow"
_DTRL = _ROOT / "DTRL-Off"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DTRL))

import Semicircle_env
import safety_gymnasium
from action_utils import unscale_action
from eenn_network import ActorEENN3Semicircle, SharedBackbone256

TASK_NAME = "Semicircle-Narrow"
ENV_ID = "SafetyPointSemicircle0-v5"
# Tunable evaluation knobs (each name is commented; see README.md).
CPU_CORE = 0                 # CPU to pin; falls back if this core is missing
WARMUP_STEPS = 1000          # env steps used to fill the latency buffer
PROFILE_STEPS = 2000         # extra profiling steps written into the buffer
N_SEEDS = 100                # number of evaluation episodes
START_SEED = 42              # first episode seed (then +1, +2, …)
DECISION_INTERVAL = 10       # control steps between exit re-selections
BUFFER_SIZE = 2000           # rolling window of measured inference latencies

P_EXIT_ESTIMATE = 95         # percentile used as an exit's predicted latency (ms)
BUDGET_LOW_TAG = "e1"        # exit that sets the lower end of the deadline band
P_BUDGET_LOW = 92            # percentile of that exit's measured latency
BUDGET_HIGH_TAG = "ef"       # exit that sets the upper end of the deadline band
P_BUDGET_HIGH = 98           # percentile of that exit's measured latency

# Success is goal-at-end with no collision (no extra return threshold).
# The narrow track makes delayed or inconsistent actions easier to expose.
SWI_ZIP = _ROOT / "SWI/runs/SWI_20260516_000551/checkpoints/SWI_1500000_steps.zip"
FULL_ZIP = _ROOT / "Full/runs/Full_20260512_115359/best_model/Full_best_model.zip"
MMS_E2_ZIP = _ROOT / "MMS/runs/MMS_20260516_012652/checkpoints/MMS_500000_steps.zip"
JOINT_MODEL = (
    _ROOT
    / "DTRL-Off/runs/DTRL-Off_20260516_034629/best_model/DTRL-Off_best_model.pt"
)

_EXITS: list[tuple[str, Path, str, int]] = [
    (
        "e1",
        _ROOT
        / "DTRL-On/runs/DTRL-On_e1_20260516_050501/best_model/DTRL-On_e1_best_model.zip",
        "ppo",
        1,
    ),
    (
        "e2",
        _ROOT
        / "DTRL-On/runs/DTRL-On_e2_20260516_053958/best_model/DTRL-On_e2_best_model.zip",
        "ppo",
        2,
    ),
    (
        "ef",
        _ROOT
        / "DTRL-On/runs/DTRL-On_full_20260516_021458/best_model/DTRL-On_full_best_model.zip",
        "ppo",
        3,
    ),
]

_MMS_EXITS: list[tuple[str, Path, str, int]] = [
    ("e1", SWI_ZIP, "ppo", 1),
    ("e2", MMS_E2_ZIP, "ppo", 2),
    ("ef", FULL_ZIP, "sac", 3),
]

_OFF_EXITS: list[tuple[str, Path, str, int]] = [
    ("e1", JOINT_MODEL, "sac", 1),
    ("e2", JOINT_MODEL, "sac", 2),
    ("ef", JOINT_MODEL, "sac", 3),
]


class SafetyToGymnasiumWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["cost"] = float(cost)
        return obs, reward, terminated, truncated, info


class PpoActor(nn.Module):
    def __init__(self, policy_net: nn.Sequential, action_net: nn.Linear) -> None:
        super().__init__()
        self.trunk = policy_net
        self.head = action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(obs))


class SacActor(nn.Module):
    def __init__(self, sac_actor: nn.Module) -> None:
        super().__init__()
        self.trunk = sac_actor.latent_pi
        self.head = sac_actor.mu

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.trunk(obs)))


class OffExit1(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone256(obs_dim)
        self.mean = nn.Linear(self.backbone.out_dim, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.backbone(obs))


class OffExit2(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone256(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.mean = nn.Linear(128, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean(self.fc3(self.fc2(self.backbone(obs))))


class OffExit3(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone256(obs_dim)
        w = self.backbone.out_dim
        self.fc2 = nn.Sequential(nn.Linear(w, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.mean = nn.Linear(64, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.fc3(self.fc2(self.backbone(obs)))
        return self.mean(self.trunk(h))


class TanhWrap(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.inner(obs))


@dataclass
class ExitActor:
    tag: str
    actor: nn.Module
    kind: str
    weight_path: Path
    depth: int


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


def _make_env() -> gym.Env:
    return SafetyToGymnasiumWrapper(safety_gymnasium.make(ENV_ID, render_mode=None))


def _goal_achieved(env: gym.Env) -> bool:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    task = getattr(base, "task", None)
    if task is not None and hasattr(task, "goal_achieved"):
        return bool(task.goal_achieved)
    if hasattr(base, "goal_achieved"):
        return bool(base.goal_achieved)
    return False


def _build_exit(tag: str, path: Path, kind: str, depth: int) -> ExitActor:
    if not path.is_file():
        raise FileNotFoundError(path)
    if kind == "ppo":
        model = PPO.load(str(path), device="cpu")
        actor = PpoActor(
            model.policy.mlp_extractor.policy_net,
            model.policy.action_net,
        ).eval()
        del model
    else:
        model = SAC.load(str(path), device="cpu")
        actor = SacActor(model.policy.actor).eval()
        del model
    return ExitActor(tag=tag, actor=actor, kind=kind, weight_path=path, depth=depth)


def _build_off_exits(obs_dim: int, act_dim: int) -> dict[str, ExitActor]:
    if not JOINT_MODEL.is_file():
        raise FileNotFoundError(JOINT_MODEL)
    ckpt = torch.load(JOINT_MODEL, map_location="cpu", weights_only=False)
    full = ActorEENN3Semicircle(obs_dim, act_dim)
    full.load_state_dict(ckpt["actor"])
    full.eval()

    specs: list[tuple[str, type[nn.Module], int, int]] = [
        ("e1", OffExit1, 1, 1),
        ("e2", OffExit2, 2, 2),
        ("ef", OffExit3, 3, 3),
    ]
    out: dict[str, ExitActor] = {}
    for tag, cls, exit_id, depth in specs:
        net: nn.Module = cls(obs_dim, act_dim)
        net.backbone.load_state_dict(full.backbone.state_dict())
        if exit_id == 1:
            net.mean.load_state_dict(full.exit1_mean.state_dict())
        elif exit_id == 2:
            net.fc2.load_state_dict(full.fc2.state_dict())
            net.fc3.load_state_dict(full.fc3.state_dict())
            net.mean.load_state_dict(full.exit2_mean.state_dict())
        else:
            net.fc2.load_state_dict(full.fc2.state_dict())
            net.fc3.load_state_dict(full.fc3.state_dict())
            net.trunk.load_state_dict(full.trunk_exit3.state_dict())
            net.mean.load_state_dict(full.exit3_mean.state_dict())
        actor = TanhWrap(net.eval()).eval()
        out[tag] = ExitActor(
            tag=tag, actor=actor, kind="sac", weight_path=JOINT_MODEL, depth=depth
        )
    return out


def _to_env_action(
    raw: np.ndarray, action_space: spaces.Box, kind: str
) -> np.ndarray:
    if kind == "sac":
        return unscale_action(raw, action_space)
    return np.clip(raw, action_space.low, action_space.high).astype(np.float32)


def _timed_exit_infer(
    exit_actor: ExitActor, obs: np.ndarray, action_space: spaces.Box
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    with torch.inference_mode():
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        scaled = exit_actor.actor(x).detach().cpu().numpy().reshape(-1)
        action = _to_env_action(scaled, action_space, exit_actor.kind)
    return action, (time.perf_counter() - t0) * 1e3


def _run_env_steps(
    timed_infer,
    buf: LatencyBuffer | None,
    seed: int,
    n_steps: int,
    *,
    record_latency: bool,
) -> None:
    env = _make_env()
    assert isinstance(env.action_space, spaces.Box)
    obs, _ = env.reset(seed=seed)
    done = False
    ep = 0
    n = 0
    while n < n_steps:
        if done:
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
            done = False
        action, lat_ms = timed_infer(obs, env.action_space)
        if record_latency and buf is not None:
            buf.push(lat_ms)
        obs, _, term, trunc, _ = env.step(action)
        done = bool(term or trunc)
        n += 1
    env.close()


def _warmup_exit_into_buffer(
    exit_actor: ExitActor,
    buf: LatencyBuffer,
    tag: str,
    seed: int,
) -> None:
    print(f"[profiling] {tag}", flush=True)
    timed = lambda obs, space, a=exit_actor: _timed_exit_infer(a, obs, space)
    _run_env_steps(timed, None, seed, WARMUP_STEPS, record_latency=False)
    _run_env_steps(timed, buf, seed + 1, PROFILE_STEPS, record_latency=True)


def _budget_tags() -> tuple[str, str]:
    by_tag = {tag: depth for tag, _, _, depth in _EXITS}
    low = BUDGET_LOW_TAG or min(by_tag, key=by_tag.get)
    high = BUDGET_HIGH_TAG or max(by_tag, key=by_tag.get)
    if low not in by_tag:
        raise SystemExit(f"BUDGET_LOW_TAG={low!r} not in _EXITS: {list(by_tag)}")
    if high not in by_tag:
        raise SystemExit(f"BUDGET_HIGH_TAG={high!r} not in _EXITS: {list(by_tag)}")
    return low, high


def _episode_success(goal_at_end: bool, had_collision: bool) -> bool:
    return bool(goal_at_end and not had_collision)


def _select_exit(
    deadline_ms: float,
    buffers: dict[str, LatencyBuffer],
    exit_specs: list[tuple[str, Path, str, int]],
) -> str:
    ordered = sorted(exit_specs, key=lambda x: x[3], reverse=True)
    for tag, _, _, _ in ordered:
        if buffers[tag].p_estimate_ms() <= deadline_ms:
            return tag
    return ordered[-1][0]


def _run_episode(
    exits: dict[str, ExitActor],
    buffers: dict[str, LatencyBuffer],
    exit_specs: list[tuple[str, Path, str, int]],
    seed: int,
    budget_low_ms: float,
    budget_high_ms: float,
) -> EpisodeStats:
    env = _make_env()
    assert isinstance(env.action_space, spaces.Box)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    ep_ret = 0.0
    step = 0
    hits = 0
    had_collision = False
    deadline_ms = budget_high_ms
    active = exit_specs[0][0]
    last_action: np.ndarray | None = None
    done = False

    while not done:
        if step % DECISION_INTERVAL == 0:
            deadline_ms = float(rng.uniform(budget_low_ms, budget_high_ms))
            active = _select_exit(deadline_ms, buffers, exit_specs)

        ex = exits[active]
        new_action, lat_ms = _timed_exit_infer(ex, obs, env.action_space)
        buffers[active].push(lat_ms)

        if lat_ms <= deadline_ms:
            hits += 1
            action = new_action
            last_action = action
        elif last_action is not None:
            action = last_action
        else:
            action = new_action
            last_action = action

        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += float(reward)
        if float(info.get("cost", 0.0)) > 0.0:
            had_collision = True
        done = bool(terminated or truncated)
        step += 1

    goal = _goal_achieved(env)
    env.close()
    return EpisodeStats(
        return_=ep_ret,
        n_steps=step,
        deadline_hits=hits,
        success=_episode_success(goal, had_collision),
    )


def _run_fixed_policy_episode(
    timed_infer,
    seed: int,
    budget_low_ms: float,
    budget_high_ms: float,
) -> EpisodeStats:
    env = _make_env()
    assert isinstance(env.action_space, spaces.Box)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)

    ep_ret = 0.0
    step = 0
    hits = 0
    had_collision = False
    deadline_ms = budget_high_ms
    last_action: np.ndarray | None = None
    done = False

    while not done:
        if step % DECISION_INTERVAL == 0:
            deadline_ms = float(rng.uniform(budget_low_ms, budget_high_ms))

        new_action, lat_ms = timed_infer(obs, env.action_space)
        if lat_ms <= deadline_ms:
            hits += 1
            action = new_action
            last_action = action
        elif last_action is not None:
            action = last_action
        else:
            action = new_action
            last_action = action

        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += float(reward)
        if float(info.get("cost", 0.0)) > 0.0:
            had_collision = True
        done = bool(terminated or truncated)
        step += 1

    goal = _goal_achieved(env)
    env.close()
    return EpisodeStats(
        return_=ep_ret,
        n_steps=step,
        deadline_hits=hits,
        success=_episode_success(goal, had_collision),
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
    print(f"summary ({N_SEEDS} seeds) — {name}:")
    print(f"  budget (ms)     = [{budget_low:.4f}, {budget_high:.4f}]")
    print(f"  mean return     = {rets.mean():.4f} ± {rets.std(ddof=0):.4f}")
    print(f"  mean hit rate   = {hit_rates.mean():.4f} ± {hit_rates.std(ddof=0):.4f}")
    print(
        f"  success rate    = {success_rate:.4f} "
        f"({sum(r.success for r in results)}/{N_SEEDS})"
    )


def _print_progress(method: str, done: int, total: int) -> None:
    print(
        f"\r[{TASK_NAME}] running {method}: {done}/{total} episodes",
        end="",
        flush=True,
    )
    if done >= total:
        print()


def _eval_dynamic(
    name: str,
    exits: dict[str, ExitActor],
    buffers: dict[str, LatencyBuffer],
    specs: list[tuple[str, Path, str, int]],
    seeds: list[int],
    budget_low: float,
    budget_high: float,
) -> list[EpisodeStats]:
    results: list[EpisodeStats] = []
    total = len(seeds)
    _print_progress(name, 0, total)
    for i, seed in enumerate(seeds, start=1):
        results.append(
            _run_episode(exits, buffers, specs, seed, budget_low, budget_high)
        )
        _print_progress(name, i, total)
    _summarize(name, results, budget_low, budget_high)
    return results


def _eval_fixed(
    name: str,
    timed_infer,
    seeds: list[int],
    budget_low: float,
    budget_high: float,
) -> list[EpisodeStats]:
    results: list[EpisodeStats] = []
    total = len(seeds)
    _print_progress(name, 0, total)
    for i, seed in enumerate(seeds, start=1):
        results.append(
            _run_fixed_policy_episode(timed_infer, seed, budget_low, budget_high)
        )
        _print_progress(name, i, total)
    _summarize(name, results, budget_low, budget_high)
    return results


def main() -> None:
    for tag, path, _, _ in _EXITS:
        if not path.is_file():
            raise SystemExit(f"missing [DTRL-ON {tag}]: {path}")
    for tag, path, _, _ in _MMS_EXITS:
        if not path.is_file():
            raise SystemExit(f"missing [MMS {tag}]: {path}")
    if not SWI_ZIP.is_file():
        raise SystemExit(f"missing [SWI]: {SWI_ZIP}")
    if not JOINT_MODEL.is_file():
        raise SystemExit(f"missing [DTRL-OFF]: {JOINT_MODEL}")

    _setup_cpu()
    env = _make_env()
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    assert isinstance(env.action_space, spaces.Box)
    env.close()

    on_exits = {
        tag: _build_exit(tag, path, kind, depth)
        for tag, path, kind, depth in _EXITS
    }
    on_buffers = {tag: LatencyBuffer() for tag, _, _, _ in _EXITS}
    mms_exits = {
        tag: _build_exit(tag, path, kind, depth)
        for tag, path, kind, depth in _MMS_EXITS
    }
    mms_buffers = {tag: LatencyBuffer() for tag, _, _, _ in _MMS_EXITS}
    off_exits = _build_off_exits(obs_dim, act_dim)
    off_buffers = {tag: LatencyBuffer() for tag, _, _, _ in _OFF_EXITS}
    swi = mms_exits["e1"]
    full = mms_exits["ef"]
    low_tag, high_tag = _budget_tags()

    print(
        f"\nDTRL-ON latency profile (real env) | "
        f"budget from {low_tag} P{P_BUDGET_LOW} / {high_tag} P{P_BUDGET_HIGH}"
    )
    for tag, _, _, _ in _EXITS:
        _warmup_exit_into_buffer(
            on_exits[tag], on_buffers[tag], f"DTRL-ON {tag}", START_SEED
        )

    budget_low = on_buffers[low_tag].percentile_ms(P_BUDGET_LOW)
    budget_high = on_buffers[high_tag].percentile_ms(P_BUDGET_HIGH)
    if budget_low > budget_high:
        budget_low, budget_high = budget_high, budget_low
    print(f"  budget (ms) = [{budget_low:.4f}, {budget_high:.4f}]\n")

    print("MMS / DTRL-OFF latency buffers (real env, for exit selection):")
    for tag, _, _, _ in _MMS_EXITS:
        _warmup_exit_into_buffer(
            mms_exits[tag], mms_buffers[tag], f"MMS {tag}", START_SEED
        )
    for tag, _, _, _ in _OFF_EXITS:
        _warmup_exit_into_buffer(
            off_exits[tag], off_buffers[tag], f"DTRL-OFF {tag}", START_SEED
        )
    _run_env_steps(
        lambda obs, space, a=swi: _timed_exit_infer(a, obs, space),
        None,
        START_SEED,
        WARMUP_STEPS,
        record_latency=False,
    )
    _run_env_steps(
        lambda obs, space, a=full: _timed_exit_infer(a, obs, space),
        None,
        START_SEED,
        WARMUP_STEPS,
        record_latency=False,
    )

    seeds = list(range(START_SEED, START_SEED + N_SEEDS))
    full_results = _eval_fixed(
        "FULL",
        lambda obs, space, a=full: _timed_exit_infer(a, obs, space),
        seeds,
        budget_low,
        budget_high,
    )
    swi_results = _eval_fixed(
        "SWI",
        lambda obs, space, a=swi: _timed_exit_infer(a, obs, space),
        seeds,
        budget_low,
        budget_high,
    )
    mms_results = _eval_dynamic(
        "MMS", mms_exits, mms_buffers, _MMS_EXITS, seeds, budget_low, budget_high
    )
    on_results = _eval_dynamic(
        "DTRL-ON", on_exits, on_buffers, _EXITS, seeds, budget_low, budget_high
    )
    off_results = _eval_dynamic(
        "DTRL-OFF", off_exits, off_buffers, _OFF_EXITS, seeds, budget_low, budget_high
    )

    print(f"\n{'=' * 70}")
    print("Final comparison (same DTRL-ON-derived budget):")
    print("  success = goal_achieved and no collision (info[cost]>0)")
    for name, results in (
        ("FULL", full_results),
        ("SWI", swi_results),
        ("MMS", mms_results),
        ("DTRL-ON", on_results),
        ("DTRL-OFF", off_results),
    ):
        rets = np.array([r.return_ for r in results], dtype=np.float64)
        hits = np.array(
            [r.deadline_hits / max(r.n_steps, 1) for r in results], dtype=np.float64
        )
        succ = float(np.mean([r.success for r in results]))
        print(
            f"  {name:8s}  return={rets.mean():8.4f}±{rets.std(ddof=0):.4f}  "
            f"hit={hits.mean():.4f}±{hits.std(ddof=0):.4f}  "
            f"success={succ:.4f}"
        )


if __name__ == "__main__":
    main()
