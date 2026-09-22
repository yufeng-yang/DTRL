"""Reproduce Figure 8 on Semicircle-Narrow.

The figure compares SWI, the first DTRL-On exit used alone, and DTRL-On
with dynamic exit selection.
"""

from envpy import ensure_gym_python

ensure_gym_python()

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "figure_and_table"
ARTIFACT = HERE / "separate/Semicircle_Narrow_artifact.py"
TABLE2 = OUT / "Table 2 Main Results.json"
N_EPISODES = 100             # DTRL-On E1-only episodes (SWI / DTRL-On come from Table 2)
START_SEED = 42              # first episode seed


@dataclass
class Row:
    """One Figure 8 method: success, return, deadline hit rate, episode count."""
    method: str
    success_rate: float
    return_mean: float
    return_std: float
    hit_mean: float
    hit_std: float
    n: int


def load_artifact():
    """Import ``Semicircle_Narrow_artifact.py`` without going through ``__main__``."""
    sys.path.insert(0, str(ARTIFACT.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location("narrow_art", ARTIFACT)
    art = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(art)
    return art


def table2_row(method):
    """Copy a Semicircle-Narrow Table 2 method into a Figure 8 ``Row``."""
    rows = json.loads(TABLE2.read_text())
    r = next(x for x in rows if x["task"] == "Semicircle-Narrow" and x["method"] == method)
    return Row(
        method,
        float(r["success_rate"]),
        float(r["return_mean"]),
        float(r["return_std"]),
        float(r["hit_mean"]),
        float(r["hit_std"]),
        100,
    )


def summarize(method, results):
    """Aggregate per-episode ``EpisodeStats`` into a Figure 8 ``Row``."""
    rets = np.array([r.return_ for r in results])
    hits = np.array([r.deadline_hits / max(r.n_steps, 1) for r in results])
    return Row(
        method,
        float(np.mean([r.success for r in results])),
        float(rets.mean()),
        float(rets.std()),
        float(hits.mean()),
        float(hits.std()),
        len(results),
    )


def eval_e1():
    """Evaluate DTRL-On Exit 1 alone under the same deadline band as Table 2."""
    art = load_artifact()
    art._setup_cpu()
    e1_best = (
        art._ROOT
        / "DTRL-On/runs/DTRL-On_e1_20260516_050501/best_model/DTRL-On_e1_best_model.zip"
    )
    ef_path = next(p for tag, p, _, _ in art._EXITS if tag == "ef")
    e1 = art._build_exit("e1", e1_best, "ppo", 1)
    ef = art._build_exit("ef", ef_path, "ppo", 3)
    buf_e1 = art.LatencyBuffer()
    buf_ef = art.LatencyBuffer()
    art._warmup_exit_into_buffer(e1, buf_e1, "DTRL-On e1", START_SEED)
    art._warmup_exit_into_buffer(ef, buf_ef, "DTRL-On ef", START_SEED)
    # Use the same measured budget range as the main evaluation.
    lo = buf_e1.percentile_ms(art.P_BUDGET_LOW)
    hi = buf_ef.percentile_ms(art.P_BUDGET_HIGH)
    if lo > hi:
        lo, hi = hi, lo
    seeds = list(range(START_SEED, START_SEED + N_EPISODES))
    results = art._eval_fixed(
        "DTRL-On E1",
        lambda obs, space, a=e1: art._timed_exit_infer(a, obs, space),
        seeds,
        lo,
        hi,
    )
    return {
        "task": "Semicircle-Narrow",
        "budget_ms": [lo, hi],
        # SWI and dynamic DTRL-On are already available from Table 2.
        "methods": [
            asdict(table2_row("SWI")),
            asdict(summarize("DTRL-On E1", results)),
            asdict(table2_row("DTRL-On")),
        ],
    }


def main():
    """Write figure8 JSON/CSV and call ``drawing/figure8.py``."""
    payload = eval_e1()
    print()
    print(f"{'Method':14s} {'Success':10s} {'Return':23s}")
    print("-" * 50)
    for row in payload["methods"]:
        print(
            f"{row['method']:14s} {100.0 * row['success_rate']:6.0f}%    "
            f"{row['return_mean']:.4f} ± {row['return_std']:.4f}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "figure8.json"
    csv_path = OUT / "figure8.csv"
    json_path.write_text(json.dumps(payload, indent=2))
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "method", "success_rate", "return_mean", "return_std", "hit_mean", "hit_std"])
        for row in payload["methods"]:
            w.writerow(
                [
                    payload["task"],
                    row["method"],
                    row["success_rate"],
                    row["return_mean"],
                    row["return_std"],
                    row["hit_mean"],
                    row["hit_std"],
                ]
            )
    print(f"\nJSON: {json_path}")
    print(f"CSV:  {csv_path}")

    sys.path.insert(0, str(ROOT / "drawing"))
    from figure8 import plot_swi_dtrlon_comparison

    print(f"PNG:  {OUT / 'figure8.png'}")
    plot_swi_dtrlon_comparison(show=True)


if __name__ == "__main__":
    main()
