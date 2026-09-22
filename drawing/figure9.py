"""Plot Figure 9: action inconsistency across computation levels.

Reads ``figure_and_table/figure9.json``.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _namedfig import named_subplots

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figure_and_table"
_DATA_JSON = _OUTPUT_DIR / "figure9.json"
_METHODS = ("MMS", "DTRL-On", "DTRL-Off")


def _load_figure9_data() -> tuple[list[str], dict[str, list[float]], dict[str, list[float]]]:
    """Return task labels and per-method inconsistency means/stds."""
    if not _DATA_JSON.is_file():
        raise FileNotFoundError(
            f"Missing {_DATA_JSON}. Run Quick_Artifact/figure9_prepard.py first."
        )
    payload = json.loads(_DATA_JSON.read_text(encoding="utf-8"))
    by_method = {row["method"]: row for row in payload["methods"]}
    inconsistency = {
        name: [float(v) for v in by_method[name]["inconsistency_mean"]]
        for name in _METHODS
    }
    inconsistency_std = {
        name: [float(v) for v in by_method[name]["inconsistency_std"]]
        for name in _METHODS
    }
    return list(payload["tasks"]), inconsistency, inconsistency_std


def plot_action_inconsistency(*, show: bool = True):
    """Grouped inconsistency bars; write ``figure9.png``."""
    tasks, inconsistency, inconsistency_std = _load_figure9_data()
    methods = list(_METHODS)

    x = np.arange(len(tasks))
    width = 0.23

    fig, ax = named_subplots(figsize=(6.8, 3.1), title="Figure 9")

    colors = {
        "MMS": "#D9D9D9",
        "DTRL-On": "#BFE8BF",
        "DTRL-Off": "#F4B6C2",
    }

    offsets = [-width, 0, width]

    for i, method in enumerate(methods):
        values = inconsistency[method]
        errors = inconsistency_std[method]

        bars = ax.bar(
            x + offsets[i],
            values,
            width=width,
            color=colors[method],
            edgecolor="black",
            linewidth=0.7,
            label=method
        )

        ax.errorbar(
            x + offsets[i],
            values,
            yerr=errors,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3
        )

        for j, (bar, val, err) in enumerate(zip(bars, values, errors)):
            label_y = val + err + 0.012
            if method == "MMS" and tasks[j] == "SC-Wide":
                label_y -= 0.012  # 0.378 The label moves slightly downwards
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylabel("Action Inconsistency", fontsize=12)
    ax.set_xlabel("Task", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=9)
    ax.tick_params(axis="y", labelsize=9)

    ax.set_ylim(0, 0.52)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        fontsize=9,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.5
    )

    ax.margins(x=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUTPUT_DIR / "figure9.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_action_inconsistency()