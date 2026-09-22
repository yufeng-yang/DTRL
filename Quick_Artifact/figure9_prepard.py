"""Reproduce Figure 9: action inconsistency across computation levels."""

from envpy import ensure_gym_python

ensure_gym_python()

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure_and_table"
TABLE2 = OUT / "Table 2 Main Results.json"

TASKS = (
    ("Pendulum", "Pendulum"),
    ("Semicircle-Wide", "SC-Wide"),
    ("Semicircle-Narrow", "SC-Narrow"),
    ("Unitree Go2 Sim", "Go2 Sim"),
)
METHODS = ("MMS", "DTRL-On", "DTRL-Off")


def main():
    """Read Table 2 inconsistency columns, write figure9 JSON/CSV, then plot."""
    rows = json.loads(TABLE2.read_text())
    # Table 2 already contains the inconsistency statistics from evaluation.
    by = {(r["task"], r["method"]): r for r in rows}
    payload = {
        "tasks": [short for _, short in TASKS],
        "methods": [
            {
                "method": method,
                "inconsistency_mean": [by[(full, method)]["inconsistency_mean"] for full, _ in TASKS],
                "inconsistency_std": [by[(full, method)]["inconsistency_std"] for full, _ in TASKS],
            }
            for method in METHODS
        ],
    }

    print(f"{'Task':12s} {'MMS':18s} {'DTRL-On':18s} {'DTRL-Off':18s}")
    print("-" * 68)
    by_method = {m["method"]: m for m in payload["methods"]}
    for i, short in enumerate(payload["tasks"]):
        cells = [
            f"{by_method[m]['inconsistency_mean'][i]:.4f} ± {by_method[m]['inconsistency_std'][i]:.4f}"
            for m in METHODS
        ]
        print(f"{short:12s} {cells[0]:18s} {cells[1]:18s} {cells[2]:18s}")

    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "figure9.json"
    csv_path = OUT / "figure9.csv"
    json_path.write_text(json.dumps(payload, indent=2))
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "short_name", "method", "inconsistency_mean", "inconsistency_std"])
        for i, (full, short) in enumerate(TASKS):
            for m in payload["methods"]:
                w.writerow([full, short, m["method"], m["inconsistency_mean"][i], m["inconsistency_std"][i]])
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")

    sys.path.insert(0, str(ROOT / "drawing"))
    from figure9 import plot_action_inconsistency

    print(f"PNG:  {OUT / 'figure9.png'}")
    plot_action_inconsistency(show=True)


if __name__ == "__main__":
    main()
