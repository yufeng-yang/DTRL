"""Re-exec Gym/Safety scripts with the sb3sg interpreter.

Cursor and some IDE Run buttons invoke ``/bin/python3`` even when the
shell prompt shows ``(sb3sg)``. That system Python does not have numpy,
matplotlib, or gymnasium.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

GYM_PY = Path(os.environ.get("DTRL_GYM_PY", sys.executable))
GO2_PY = Path(os.environ.get("DTRL_GO2_PY", sys.executable))


def ensure_gym_python() -> None:
    """Replace this process with ``GYM_PY`` when launched by another interpreter."""
    target = GYM_PY.resolve()
    if Path(sys.executable).resolve() == target:
        return
    os.execv(str(target), [str(target), *sys.argv])
