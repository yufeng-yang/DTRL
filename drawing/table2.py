"""Plot Table 2 as a booktabs-style PNG from the shipped JSON.

Reads ``figure_and_table/Table 2 Main Results.json``. Does not evaluate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from _booktabs import draw_booktabs
from _namedfig import named_subplots

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figure_and_table"
_TABLE2_JSON = _OUTPUT_DIR / "Table 2 Main Results.json"
_METHODS = ("Full", "SWI", "MMS", "DTRL-On", "DTRL-Off")
_TASKS = (
    "Pendulum",
    "Semicircle-Wide",
    "Semicircle-Narrow",
    "Unitree Go2 Sim",
)
_HEADERS = (
    "Task",
    "Method",
    "Return",
    "Deadline Hit Rate",
    "Success Rate",
    "Inconsistency",
)


def _load_table2() -> dict[tuple[str, str], dict]:
    if not _TABLE2_JSON.is_file():
        raise FileNotFoundError(
            f"Missing {_TABLE2_JSON}. Run Quick_Artifact/Table2_MainResults.py first."
        )
    rows = json.loads(_TABLE2_JSON.read_text(encoding="utf-8"))
    by_key = {(row["task"], row["method"]): row for row in rows}
    missing = [
        (task, method)
        for task in _TASKS
        for method in _METHODS
        if (task, method) not in by_key
    ]
    if missing:
        raise KeyError(f"Table 2 missing rows: {missing}")
    return by_key


def _fmt_mean_std(mean, std) -> str:
    return f"{float(mean):.4f} ± {float(std):.4f}"


def plot_table2(*, show: bool = True) -> None:
    """Draw Table 2 and save ``Table 2 Main Results.png``."""
    by_key = _load_table2()
    cells: list[list[str]] = []
    bold_row: list[bool] = []
    for task in _TASKS:
        for method in _METHODS:
            row = by_key[(task, method)]
            if row.get("inconsistency_mean") is None:
                inc = "–"
            else:
                inc = _fmt_mean_std(row["inconsistency_mean"], row["inconsistency_std"])
            cells.append(
                [
                    "",
                    method,
                    _fmt_mean_std(row["return_mean"], row["return_std"]),
                    _fmt_mean_std(row["hit_mean"], row["hit_std"]),
                    f"{100.0 * float(row['success_rate']):.0f}%",
                    inc,
                ]
            )
            bold_row.append(method in ("DTRL-On", "DTRL-Off"))

    n_m = len(_METHODS)
    span_col0 = [
        (i * n_m, i * n_m + n_m - 1, task) for i, task in enumerate(_TASKS)
    ]
    group_after = [n_m * (i + 1) - 1 for i in range(len(_TASKS) - 1)]

    fig, ax = named_subplots(figsize=(11.4, 5.6), title="Table 2")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
    ax.set_title("Table 2  Main Results", fontsize=13, fontweight="bold", pad=6)
    draw_booktabs(
        ax,
        list(_HEADERS),
        cells,
        col_x=[0.03, 0.22, 0.40, 0.60, 0.76, 0.90],
        col_align=["left", "left", "center", "center", "center", "center"],
        bold_row=bold_row,
        group_after=group_after,
        span_col0=span_col0,
        fontsize=9.5,
    )
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "Table 2 Main Results.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"PNG: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_table2(show="--no-show" not in sys.argv)
