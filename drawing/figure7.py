"""Plot Figure 7: Full-model success with and without dynamic deadlines.

Needs ``figure7_full_compare.json`` (without deadline) and
``Table 2 Main Results.json`` (with deadline).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _namedfig import named_subplots

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figure_and_table"
_WITHOUT_DEADLINE_JSON = _OUTPUT_DIR / "figure7_full_compare.json"
_TABLE2_JSON = _OUTPUT_DIR / "Table 2 Main Results.json"

_TASKS = (
    ("Pendulum", "Pendulum"),
    ("Semicircle-Wide", "SC-Wide"),
    ("Semicircle-Narrow", "SC-Narrow"),
    ("Unitree Go2 Sim", "Go2 Sim"),
)


def _as_percent(rate: float) -> int:
    """Convert a success rate in [0, 1] to a rounded percentage."""
    return int(round(100.0 * float(rate)))


def _load_without_deadline() -> list[int]:
    """Success percentages for Full evaluated with an infinite deadline."""
    rows = json.loads(_WITHOUT_DEADLINE_JSON.read_text(encoding="utf-8"))
    by_task = {row["task"]: row["without_deadline"]["success_rate"] for row in rows}
    return [_as_percent(by_task[full_name]) for full_name, _ in _TASKS]


def _load_with_deadline() -> list[int]:
    """Success percentages for Full from Table 2 (dynamic deadline)."""
    if not _TABLE2_JSON.is_file():
        raise FileNotFoundError(
            f"Missing {_TABLE2_JSON}. Run Quick_Artifact/Table2_MainResults.py first."
        )
    rows = json.loads(_TABLE2_JSON.read_text(encoding="utf-8"))
    by_task = {
        row["task"]: float(row["success_rate"])
        for row in rows
        if row["method"] == "Full"
    }
    return [_as_percent(by_task[full_name]) for full_name, _ in _TASKS]


def plot_full_success_comparison(*, show: bool = True):
    """Grouped bars per task; write ``figure7.png``."""
    tasks = [short_name for _, short_name in _TASKS]
    no_deadline_success = _load_without_deadline()
    dynamic_success = _load_with_deadline()

    x = np.arange(len(tasks))
    width = 0.28

    # Raise the entire chart to make the histogram area higher
    fig, ax = named_subplots(figsize=(6.6, 3.7), title="Figure 7")

    success_color = "#BFE8BF"

    # Left bar: without deadline, translucent
    ax.bar(
        x - width / 2,
        no_deadline_success,
        width=width,
        color=success_color,
        alpha=0.35,
        edgecolor="black",
        linewidth=0.7,
        label="Without Deadline"
    )

    # Right bar: with dynamic deadline, solid
    ax.bar(
        x + width / 2,
        dynamic_success,
        width=width,
        color=success_color,
        alpha=1.0,
        edgecolor="black",
        linewidth=0.7,
        label="With Dynamic Deadline"
    )

    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_xlabel("Task", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)

    # Leave space for numbers above 100, but scale only to 100
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])

    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Reduce left and right white space
    ax.margins(x=0.035)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=2,
        fontsize=10,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5
    )

    # Value labels
    for xpos, val in zip(x - width / 2, no_deadline_success):
        ax.text(
            xpos,
            val + 4,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False
        )

    for xpos, val in zip(x + width / 2, dynamic_success):
        label_y = val + 4 if val > 0 else 4
        ax.text(
            xpos,
            label_y,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=False
        )

    # Leave space for the legend above, but do not compress the histogram area too much
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUTPUT_DIR / "figure7.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_full_success_comparison(show="--no-show" not in sys.argv)