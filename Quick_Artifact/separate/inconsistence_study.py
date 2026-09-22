"""Measure cross-exit action inconsistency for Table 2 and Figure 9."""

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QA = Path(__file__).resolve().parents[1]
if str(QA) not in sys.path:
    sys.path.insert(0, str(QA))
from relocate import relocate_module  # noqa: E402

SRC = ROOT / "Src_Training"
GYM_PY = Path(os.environ.get("DTRL_GYM_PY", sys.executable))
GO2_PY = Path(os.environ.get("DTRL_GO2_PY", sys.executable))
MARKER = "__INCONSISTENCY_RESULT__="
START_SEED = 42              # first episode seed for inconsistency rollouts
WORKERS = 8                  # parallel workers for Gym/Safety inconsistency
GO2_STEPS = 500              # Go2 steps used to estimate inconsistency


@dataclass
class Task:
    """One inconsistency worker: key, paper task name, source file, Python binary."""
    key: str
    name: str
    source: Path
    python: Path


TASKS = {
    "pendulum": Task(
        "pendulum",
        "Pendulum",
        SRC / "Pendulum/evaluation/inconsistance_study.py",
        GYM_PY,
    ),
    "semicircle-wide": Task(
        "semicircle-wide",
        "Semicircle-Wide",
        SRC / "Semicircle-Wide/evaluation/inconsistancy_study.py",
        GYM_PY,
    ),
    "semicircle-narrow": Task(
        "semicircle-narrow",
        "Semicircle-Narrow",
        SRC / "Semicircle-Narrow/evaluation/inconsistancy_study.py",
        GYM_PY,
    ),
    "unitree-go2": Task(
        "unitree-go2",
        "Unitree Go2 Sim",
        SRC / "Unitree Go2 Sim/evaluation/inconsistancy_study.py",
        GO2_PY,
    ),
}


def load_module(path, name):
    """Import a per-task ``inconsistancy_study.py`` (or Go2 equivalent)."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def row(task, stats):
    """Normalize one method’s inconsistency stats into a printable dict."""
    method = str(stats["method"]).replace("DTRL_ON", "DTRL-On").replace("DTRL_OFF", "DTRL-Off")
    return {
        "task": task,
        "method": method,
        "mean": float(stats["inconsistency_mean"]),
        "std": float(stats["inconsistency_std"]),
        "p95": float(stats["inconsistency_p95"]),
    }


def run_gym(task):
    """Compute MMS / DTRL-On / DTRL-Off inconsistency on a Gym or Safety-Gym task."""
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces

    mod = load_module(task.source, f"_inc_{task.key.replace('-', '_')}")
    relocate_module(mod, ROOT)

    env = mod._make_env(mod.ENV_ID) if hasattr(mod, "_make_env") else gym.make(mod.ENV_ID)
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        action_space = env.action_space
    finally:
        env.close()

    seeds = list(range(START_SEED, START_SEED + int(mod.N_EPISODES)))
    suites = [mod.MmsSuite(), mod.DtrlOnSuite(), mod.DtrlOffSuite(obs_dim, act_dim)]
    rows = []
    for suite in suites:
        stats = mod.compute_method_inconsistency(
            suite, action_space, seeds, obs_dim, act_dim, n_workers=WORKERS
        )
        rows.append(row(task.name, stats))
    return rows


def run_go2(task):
    """Compute the same three methods on the Genesis Go2 environment (CPU)."""
    import torch

    if "mms_eval" not in sys.modules:
        mms = load_module(task.source.parent / "MMS_eval.py", "mms_eval")
        relocate_module(mms, ROOT)
    mod = load_module(task.source, "_inconsistency_unitree_go2")
    relocate_module(mod, ROOT)
    device = torch.device("cpu")
    specs = [
        ("MMS", True, mod._create_mms_env, mod._load_mms_policies),
        ("DTRL_ON", True, mod._create_mms_env, mod._load_dtrl_on_policies),
        ("DTRL_OFF", False, mod._create_env, mod._load_dtrl_off_policies),
    ]
    rows = []
    for method, random_vx, env_factory, policy_loader in specs:
        mod._init_genesis(use_cpu=True, gs_seed=mod.GS_SEED)
        env = env_factory()
        try:
            action_range = float(env.env_cfg["action_range"])
            policies = policy_loader(device, action_range)
            stats = mod.compute_method_inconsistency(
                method,
                policies,
                env,
                steps_per_exit=GO2_STEPS,
                seed_start=START_SEED,
                random_vx=random_vx,
            )
            rows.append(row(task.name, stats))
        finally:
            mod._genesis_teardown()
    return rows


def run_task(task):
    """Run ``--worker`` in the task’s conda Python so Gym and Genesis stay isolated."""
    # Keep Genesis and Gym dependencies isolated in their own environments.
    print(f"[{task.name}] running...", flush=True)
    cmd = [str(task.python), str(Path(__file__).resolve()), "--worker", task.key]
    p = subprocess.run(cmd, text=True, capture_output=True)
    marker = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith(MARKER)), None)
    if p.returncode != 0 or marker is None:
        raise RuntimeError(p.stdout + p.stderr)
    return json.loads(marker[len(MARKER) :])


def main():
    """Print the inconsistency table, or dump JSON when invoked as ``--worker``."""
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        task = TASKS[sys.argv[2]]
        rows = run_go2(task) if task.key == "unitree-go2" else run_gym(task)
        print(MARKER + json.dumps(rows))
        return

    rows = []
    for task in TASKS.values():
        rows.extend(run_task(task))
    print()
    print(f"{'Task':22s} {'Method':10s} {'Inconsistency':24s}")
    print("-" * 58)
    for r in rows:
        print(f"{r['task']:22s} {r['method']:10s} {r['mean']:.4f} ± {r['std']:.4f}")


if __name__ == "__main__":
    main()
