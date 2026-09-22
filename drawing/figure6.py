"""Plot Figure 6: deadline hit rate versus task success rate.

Reads ``figure_and_table/Table 2 Main Results.json``. Does not evaluate.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from _namedfig import named_subplots

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figure_and_table"
_TABLE2_JSON = _OUTPUT_DIR / "Table 2 Main Results.json"
_METHODS = ("Full", "SWI", "MMS", "DTRL-On", "DTRL-Off")
_TASKS = (
    ("Pendulum", "Pendulum"),
    ("Semicircle-Wide", "Semicircle-Wide"),
    ("Semicircle-Narrow", "Semicircle-Narrow"),
    ("Unitree Go2 Sim", "Unitree Go2 Simulation"),
)
_COLORS = {
    "Full": "#1f77b4",
    "SWI": "#ff7f0e",
    "MMS": "#2ca02c",
    "DTRL-On": "#d62728",
    "DTRL-Off": "#9467bd",
}


def _load_table2() -> dict[tuple[str, str], dict]:
    """Index Table 2 JSON by (task, method). Raise if a required cell is missing."""
    if not _TABLE2_JSON.is_file():
        raise FileNotFoundError(
            f"Missing {_TABLE2_JSON}. Run Quick_Artifact/Table2_MainResults.py first."
        )
    rows = json.loads(_TABLE2_JSON.read_text(encoding="utf-8"))
    by_key = {(row["task"], row["method"]): row for row in rows}
    missing = [
        (task, method)
        for task, _ in _TASKS
        for method in _METHODS
        if (task, method) not in by_key
    ]
    if missing:
        raise KeyError(f"Table 2 missing rows: {missing}")
    return by_key


def plot_tradeoff(*, show: bool = True) -> None:
    """Draw the 2×2 scatter and save ``figure6.png``."""
    by_key = _load_table2()
    fig, axes = named_subplots(2, 2, figsize=(8.6, 6.4), title="Figure 6")
    handles = []
    labels = []

    for ax, (task, title), letter in zip(
        axes.ravel(),
        _TASKS,
        "abcd",
    ):
        hit = [float(by_key[(task, method)]["hit_mean"]) for method in _METHODS]
        success = [
            100.0 * float(by_key[(task, method)]["success_rate"])
            for method in _METHODS
        ]
        for method, x, y in zip(_METHODS, hit, success):
            scatter = ax.scatter(
                x,
                y,
                s=70,
                color=_COLORS[method],
                label=method,
                zorder=3,
            )
            if ax is axes[0, 0]:
                handles.append(scatter)
                labels.append(method)

        ax.set_xlabel("Deadline Hit Rate", fontweight="bold")
        ax.set_ylabel("Success Rate (%)", fontweight="bold")
        ax.set_title(f"({letter}) {title}")
        hit_lo = min(hit)
        ax.set_xlim(max(0.0, hit_lo - 0.12), 1.02)
        ax.set_ylim(-5, 108)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(_METHODS),
        prop={"size": 8, "weight": "bold"},
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUTPUT_DIR / "figure6.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_tradeoff()
