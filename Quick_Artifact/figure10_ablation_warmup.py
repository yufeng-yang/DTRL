"""Reproduce Figure 10 from the shipped TensorBoard evaluation logs.
"""

from envpy import GYM_PY, ensure_gym_python

ensure_gym_python()

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        [str(GYM_PY), str(ROOT / "drawing/figure10.py"), "--no-show"],
        check=True,
        env=env,
    )
