"""Reproduce Figure 7: Full-model success with and without dynamic deadlines."""

import csv
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from relocate import relocate_module

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Src_Training"
OUT = ROOT / "figure_and_table"
GYM_PY = Path(os.environ.get("DTRL_GYM_PY", sys.executable))
GO2_PY = Path(os.environ.get("DTRL_GO2_PY", sys.executable))
N_EPISODES = 100             # Full-without-deadline episodes per task
START_SEED = 42              # first episode seed
MARKER = "__FIGURE7_RESULT__="


@dataclass
class Task:
    """One Full-eval backend: key, paper name, plot label, source file, Python binary."""
    key: str
    name: str
    short_name: str
    source: Path
    python: Path


TASKS = {
    "pendulum": Task(
        "pendulum",
        "Pendulum",
        "Pendulum",
        SRC / "Pendulum/evaluation/Full_eval.py",
        GYM_PY,
    ),
    "semicircle-wide": Task(
        "semicircle-wide",
        "Semicircle-Wide",
        "SC-Wide",
        SRC / "Semicircle-Wide/evaluation/Full_eval.py",
        GYM_PY,
    ),
    "semicircle-narrow": Task(
        "semicircle-narrow",
        "Semicircle-Narrow",
        "SC-Narrow",
        SRC / "Semicircle-Narrow/evaluation/Full_eval.py",
        GYM_PY,
    ),
    "unitree-go2": Task(
        "unitree-go2",
        "Unitree Go2 Sim",
        "Go2 Sim",
        SRC / "Unitree Go2 Sim/evaluation/Full_eval.py",
        GO2_PY,
    ),
}


def load_module(path, name):
    """Import ``path`` as ``name`` so Figure 7 can call that evaluator’s helpers."""
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fix_paths(mod):
    """Rewrite leftover training-machine absolute paths onto this repository."""
    relocate_module(mod, ROOT)


def summarize(results):
    """Mean/std return, hit rate, and success from a list of episode stats."""
    import numpy as np

    rets = np.array([r.return_ for r in results])
    hits = np.array([r.deadline_hits / max(r.n_steps, 1) for r in results])
    return {
        "success_rate": float(np.mean([r.success for r in results])),
        "return_mean": float(rets.mean()),
        "return_std": float(rets.std()),
        "hit_mean": float(hits.mean()),
        "hit_std": float(hits.std()),
        "n": len(results),
    }


def bind_cpu(mod):
    """Pin to ``CPU_CORE`` when that core exists; otherwise the first allowed core."""
    if hasattr(os, "sched_setaffinity") and hasattr(mod, "CPU_CORE"):
        available = sorted(os.sched_getaffinity(0))
        wanted = int(mod.CPU_CORE)
        if available and wanted not in available:
            mod.CPU_CORE = available[0]
    mod._setup_cpu()


def eval_task(task):
    """Run the Full policy of ``task`` with an effectively infinite deadline."""
    mod = load_module(task.source, f"_fig7_{task.key.replace('-', '_')}")
    fix_paths(mod)
    # A very large deadline leaves the policy unconstrained without changing
    # the task-specific evaluation code.
    mod.DEADLINE_LOW_MS = 1e9
    mod.DEADLINE_HIGH_MS = 1e9
    print(f"[{task.name}] loading Full model", flush=True)
    ctx = None
    try:
        if task.key == "pendulum":
            import gymnasium as gym
            import numpy as np
            from gymnasium import spaces

            bind_cpu(mod)
            ckpt = mod.torch.load(mod.JOINT_MODEL, map_location="cpu", weights_only=False)
            env = gym.make(mod.ENV_ID)
            obs_dim = int(np.prod(env.observation_space.shape))
            act_dim = int(np.prod(env.action_space.shape))
            act_space = env.action_space
            env.close()
            full = mod.ActorEENN3Deep(obs_dim, act_dim)
            full.load_state_dict(ckpt["actor"])
            actor = mod._build_ef(full, obs_dim, act_dim)
            results = []
            for i, seed in enumerate(range(START_SEED, START_SEED + N_EPISODES)):
                results.append(mod._run_episode(actor, act_space, seed))
                print(f"[{task.name}] Without Deadline {i + 1}/{N_EPISODES}", flush=True)
        elif task.key == "unitree-go2":
            import torch

            bind_cpu(mod)
            device = torch.device("cpu")
            actor = mod._load_full_actor(mod.DEFAULT_MODEL, device)
            mod._init_genesis(True, mod.GS_SEED)
            import genesis as gs

            env = mod._create_env()
            fixed = torch.tensor(
                [[mod.VX_CMD, mod.VY_CMD, mod.YAW_CMD]],
                dtype=gs.tc_float,
                device=gs.device,
            )
            mod._pin_commands(env, fixed)
            ctx = env
            rng = __import__("numpy").random.default_rng(START_SEED)
            buf = mod.LatencyBuffer()
            results = []
            for ep in range(N_EPISODES):
                results.append(
                    mod._run_episode(env, actor, buf, fixed=fixed, device=device, ep_idx=ep, rng=rng)
                )
                print(f"[{task.name}] Without Deadline {ep + 1}/{N_EPISODES}", flush=True)
        else:
            bind_cpu(mod)
            actor, env_id = mod._build_actor(mod.MODEL_ZIP)
            env = mod._make_env(env_id)
            act_space = env.action_space
            env.close()
            results = []
            for i, seed in enumerate(range(START_SEED, START_SEED + N_EPISODES)):
                results.append(mod._run_episode(actor, env_id, act_space, seed))
                print(f"[{task.name}] Without Deadline {i + 1}/{N_EPISODES}", flush=True)
        summary = summarize(results)
    finally:
        if task.key == "unitree-go2":
            mod._genesis_teardown()
    return {"task": task.name, "short_name": task.short_name, "without_deadline": summary}


def run_task(task):
    """Spawn the task in its own interpreter and parse the ``__FIGURE7_RESULT__`` marker."""
    # Go2 and Gym tasks live in separate Python environments.
    cmd = [str(task.python), str(Path(__file__).resolve()), "--worker", task.key]
    p = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for line in p.stdout:
        print(line, end="")
        lines.append(line)
    text = "".join(lines)
    if p.wait() != 0:
        raise RuntimeError(text)
    marker = [ln for ln in text.splitlines() if ln.startswith(MARKER)][-1]
    return json.loads(marker[len(MARKER) :])


def main():
    """Evaluate all tasks, write figure7 JSON/CSV, then plot."""
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        print(MARKER + json.dumps(eval_task(TASKS[sys.argv[2]])))
        return

    rows = [run_task(TASKS[key]) for key in TASKS]
    print()
    print(f"{'Task':22s} {'Success':10s} {'Return':23s}")
    print("-" * 58)
    for row in rows:
        r = row["without_deadline"]
        print(
            f"{row['task']:22s} {100.0 * r['success_rate']:6.0f}%    "
            f"{r['return_mean']:.4f} ± {r['return_std']:.4f}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "figure7_full_compare.json"
    csv_path = OUT / "figure7_full_compare.csv"
    json_path.write_text(json.dumps(rows, indent=2))
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "short_name", "success_rate", "return_mean", "return_std"])
        for row in rows:
            r = row["without_deadline"]
            w.writerow([row["task"], row["short_name"], r["success_rate"], r["return_mean"], r["return_std"]])
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")

    print(f"PNG:  {OUT / 'figure7.png'}")
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [str(GYM_PY), str(ROOT / "drawing/figure7.py"), "--no-show"],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
