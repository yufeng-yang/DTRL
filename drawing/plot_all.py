"""Redraw every shipped table/figure PNG from JSON and TensorBoard logs.

Does not evaluate policies. Output names match the paper elements:

    figure_and_table/Table 2 Main Results.png
    figure_and_table/Table 3 Ablation2.png
    figure_and_table/figure6.png
    figure_and_table/figure7.png
    figure_and_table/figure8.png
    figure_and_table/figure9.png
    figure_and_table/figure10.png

Figures are saved as they are drawn. On a machine with a display they also
open in windows (close them to exit). Use ``--no-show`` to save only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _want_display() -> bool:
    if "--no-show" in sys.argv:
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# INSTRUCTIONS often sets MPLBACKEND=Agg; drop it when we actually have a screen.
if _want_display():
    os.environ.pop("MPLBACKEND", None)
else:
    os.environ.setdefault("MPLBACKEND", "Agg")

from figure6 import plot_tradeoff
from figure7 import plot_full_success_comparison
from figure8 import plot_swi_dtrlon_comparison
from figure9 import plot_action_inconsistency
from figure10 import plot_full_exit_warmup
from table2 import plot_table2
from table3 import plot_table3
import matplotlib.pyplot as plt

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figure_and_table"

# Labels match drawing/ script names (table2, table3, figure6–10), not 1…N.
_JOBS = (
    ("Table 2", "Table 2 Main Results.png", plot_table2),
    ("Table 3", "Table 3 Ablation2.png", plot_table3),
    ("Figure 6", "figure6.png", plot_tradeoff),
    ("Figure 7", "figure7.png", plot_full_success_comparison),
    ("Figure 8", "figure8.png", plot_swi_dtrlon_comparison),
    ("Figure 9", "figure9.png", plot_action_inconsistency),
    ("Figure 10", "figure10.png", plot_full_exit_warmup),
)


def main() -> None:
    """Save each PNG and, when possible, show all figure windows together."""
    display = _want_display()
    if display:
        _real_show = plt.show

        def _nonblock(*args, **kwargs):
            kwargs["block"] = False
            return _real_show(*args, **kwargs)

        plt.show = _nonblock

    for label, name, fn in _JOBS:
        print(f"[plot_all] {label}  →  {name}", flush=True)
        fn(show=display)
        print(f"           {_OUTPUT_DIR / name}", flush=True)

    if display:
        plt.show = _real_show
        print("[plot_all] close the figure windows to exit", flush=True)
        plt.show()


if __name__ == "__main__":
    main()
