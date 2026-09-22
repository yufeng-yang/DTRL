"""Plot Figure 8: Semicircle-Narrow SWI vs DTRL-On E1 vs dynamic DTRL-On.

Reads ``figure_and_table/figure8.json``.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _namedfig import named_subplots

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figure_and_table"
_DATA_JSON = _OUTPUT_DIR / "figure8.json"
_METHODS = ("SWI", "DTRL-On E1", "DTRL-On")


def _aligned_label_ys(ax1, ax2, x1, y1, x2, y2, *, pad_px: float = 5):
    """The two-axis column top labels are aligned to the same screen height without changing the column and y-axis range."""
    p1 = ax1.transData.transform((x1, y1))
    p2 = ax2.transData.transform((x2, y2))
    y_disp = max(p1[1], p2[1]) + pad_px
    return (
        ax1.transData.inverted().transform((p1[0], y_disp))[1],
        ax2.transData.inverted().transform((p2[0], y_disp))[1],
    )


def _load_figure8_data() -> tuple[list[str], list[float], list[float], list[int]]:
    """Return method names, mean returns, return stds, and success percentages."""
    if not _DATA_JSON.is_file():
        raise FileNotFoundError(
            f"Missing {_DATA_JSON}. Run Quick_Artifact/figure8_prepare.py first."
        )
    payload = json.loads(_DATA_JSON.read_text(encoding="utf-8"))
    by_method = {row["method"]: row for row in payload["methods"]}
    methods = list(_METHODS)
    returns = [float(by_method[name]["return_mean"]) for name in methods]
    return_std = [float(by_method[name]["return_std"]) for name in methods]
    success = [
        int(round(100.0 * float(by_method[name]["success_rate"])))
        for name in methods
    ]
    return methods, returns, return_std, success


def plot_swi_dtrlon_comparison(*, show: bool = True):
    """Grouped return/success bars; write ``figure8.png``."""
    methods, returns, return_std, success = _load_figure8_data()

    x_gap = 0.72  # The three groups of methods are closer together
    x = np.arange(len(methods)) * x_gap
    bar_width = 0.26
    group_half = bar_width / 2
    x_margin = 0.32
    align_idx = 2  # DTRL-On: Align 100 and 8.05 dimensions

    success_color = "#BFE8BF"
    return_color = "#F4B6C2"

    fig, ax1 = named_subplots(figsize=(4.0, 2.1), title="Figure 8")
    ax2 = ax1.twinx()

    ax1.bar(
        x - group_half,
        success,
        width=bar_width,
        color=success_color,
        edgecolor="black",
        linewidth=0.8,
    )

    bars_ret = ax2.bar(
        x + group_half,
        returns,
        width=bar_width,
        color=return_color,
        edgecolor="black",
        linewidth=0.8,
    )
    ax2.errorbar(
        x + group_half,
        returns,
        yerr=return_std,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=4,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylabel("Success Rate (%)", fontsize=10)
    ax2.set_ylabel("Return", fontsize=10)
    ax1.set_xlim(-x_margin, x[-1] + x_margin)
    ax1.margins(x=0)

    ax1.set_ylim(0, 112)
    ax1.set_yticks([0, 50, 100])
    ax2.set_ylim(0, max(9.6, max(r + s for r, s in zip(returns, return_std)) + 1.2))

    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.tick_params(axis="x", labelsize=9)
    ax2.tick_params(axis="y", labelsize=9)

    for i, bar_s in enumerate(ax1.patches):
        val_s = success[i]
        xc_s = bar_s.get_x() + bar_s.get_width() / 2
        y_s = val_s + 3 if val_s > 0 else 3

        bar_r = bars_ret[i]
        val_r = returns[i]
        err_r = return_std[i]
        xc_r = bar_r.get_x() + bar_r.get_width() / 2
        y_r = val_r + err_r + 0.18

        if i == align_idx:
            y_s, y_r = _aligned_label_ys(
                ax1, ax2, xc_s, y_s, xc_r, y_r, pad_px=-2
            )

        ax1.text(xc_s, y_s, f"{val_s}", ha="center", va="bottom", fontsize=9)
        ax2.text(xc_r, y_r, f"{val_r:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_OUTPUT_DIR / "figure8.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_swi_dtrlon_comparison()
