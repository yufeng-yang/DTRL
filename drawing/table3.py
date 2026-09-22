"""Plot Table 3 as a booktabs-style PNG from the shipped JSON/CSV.

Reads ``figure_and_table/Table 3 Ablation2.json`` when present, otherwise
the CSV. Does not evaluate.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from _booktabs import draw_booktabs
from _namedfig import named_subplots

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figure_and_table"
_JSON = _OUTPUT_DIR / "Table 3 Ablation2.json"
_CSV = _OUTPUT_DIR / "Table 3 Ablation2.csv"
_HEADERS = ("Loss Weights (E1 : E2 : E3)", "E1", "E2", "E3")


def _load_rows() -> list[tuple[str, str, str, str]]:
    if _JSON.is_file():
        payload = json.loads(_JSON.read_text(encoding="utf-8"))
        return [
            (
                str(row["loss_weights"]),
                f"{float(row['e1']):.2f}",
                f"{float(row['e2']):.2f}",
                f"{float(row['e3']):.2f}",
            )
            for row in payload
        ]
    if not _CSV.is_file():
        raise FileNotFoundError(
            f"Missing {_JSON} and {_CSV}. Run Quick_Artifact/table3_Ablation2.py first."
        )
    with _CSV.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            (row["Loss Weights (E1 : E2 : E3)"], row["E1"], row["E2"], row["E3"])
            for row in reader
        ]


def plot_table3(*, show: bool = True) -> None:
    """Draw Table 3 and save ``Table 3 Ablation2.png``."""
    raw = _load_rows()
    cells = [list(row) for row in raw]
    fig, ax = named_subplots(figsize=(8.2, 2.15), title="Table 3")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.78, bottom=0.08)
    ax.set_title("Table 3  Loss-Weight Allocation", fontsize=13, fontweight="bold", pad=4)
    draw_booktabs(
        ax,
        list(_HEADERS),
        cells,
        col_x=[0.28, 0.58, 0.74, 0.90],
        col_align=["center", "center", "center", "center"],
        fontsize=10,
    )
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUTPUT_DIR / "Table 3 Ablation2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"PNG: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    plot_table3(show="--no-show" not in sys.argv)
