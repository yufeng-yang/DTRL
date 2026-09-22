"""Plot Figure 10: Pendulum full-exit warm-up ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from _namedfig import named_subplots


WARMUP_STEPS = 30_000
SMOOTHING = 0.5
SCALAR_TAGS = (
    "reward_eval/exit1_mean_return",
    "reward_eval/exit2_mean_return",
    "reward_eval/exit3_mean_return",
)

_CODE = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _CODE / "figure_and_table"
_RUN_ROOT = (
    _CODE / "Src_Training" / "Ablation Study" / "Full-Exit Warm-up" / "runs"
)


def _latest_run(prefix: str) -> Path:
    """Newest TensorBoard run directory whose name starts with ``prefix``."""
    events = [
        event
        for event in _RUN_ROOT.rglob("events.out.tfevents.*")
        if event.parent.name.startswith(prefix)
    ]
    if not events:
        raise FileNotFoundError(f"No TensorBoard run beginning with {prefix!r}")
    return max(events, key=lambda event: event.stat().st_mtime).parent


def _smooth(values: np.ndarray, factor: float) -> np.ndarray:
    """Exponential moving average used to match the paper’s curve smoothing."""
    if values.size == 0 or factor <= 0.0:
        return values
    result = np.empty_like(values, dtype=np.float64)
    result[0] = values[0]
    for index in range(1, values.size):
        result[index] = factor * result[index - 1] + (1.0 - factor) * values[index]
    return result


def _scalar(run: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one TensorBoard scalar and return (steps, smoothed values)."""
    accumulator = EventAccumulator(str(run), size_guidance={"scalars": 0})
    accumulator.Reload()
    by_step = {
        int(event.step): float(event.value) for event in accumulator.Scalars(tag)
    }
    steps = np.asarray(sorted(by_step), dtype=np.int64)
    values = np.asarray([by_step[int(step)] for step in steps], dtype=np.float64)
    return steps, _smooth(values, SMOOTHING)


def plot_full_exit_warmup(*, show: bool = True) -> None:
    """Three-exit training curves with vs without warm-up; write ``figure10.png``."""
    with_warmup = _latest_run("full_exit_with_warmup_")
    without_warmup = _latest_run("full_exit_without_warmup_")
    fig, axes = named_subplots(1, 3, figsize=(12.2, 3.4), sharey=True, title="Figure 10")
    colors = {"with": "#1f77b4", "without": "#ff7f0e"}

    for exit_index, (ax, tag) in enumerate(zip(axes, SCALAR_TAGS), start=1):
        steps_with, values_with = _scalar(with_warmup, tag)
        steps_without, values_without = _scalar(without_warmup, tag)
        ax.plot(
            steps_with,
            values_with,
            color=colors["with"],
            linewidth=1.5,
            label="With warm-up",
        )
        ax.plot(
            steps_without,
            values_without,
            color=colors["without"],
            linewidth=1.5,
            label="Without warm-up",
        )
        ax.axvline(
            WARMUP_STEPS,
            color="#6ba3bf",
            linestyle="--",
            linewidth=1.2,
            label="Warm-up ends",
        )
        ax.set_title(f"Exit {exit_index}")
        ax.set_xlabel("Training Steps")
        ax.set_xlim(0, 60_000)
        ax.grid(True, alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean Return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = _OUTPUT_DIR / "figure10.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"PNG: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_full_exit_warmup(show="--no-show" not in sys.argv)
