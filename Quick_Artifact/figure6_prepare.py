"""Reproduce Figure 6: deadline hit rate versus task success rate."""

from envpy import ensure_gym_python

ensure_gym_python()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Figure 6 is drawn directly from the Table 2 results.
sys.path.insert(0, str(ROOT / "drawing"))
from figure6 import plot_tradeoff

if __name__ == "__main__":
    plot_tradeoff()
