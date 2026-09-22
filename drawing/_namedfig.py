"""Create a matplotlib figure whose window title matches the paper label."""

from __future__ import annotations

import matplotlib.pyplot as plt


def named_subplots(*args, title: str, **kwargs):
    """``plt.subplots`` with ``num=title`` and a matching window title."""
    fig, ax = plt.subplots(*args, num=title, **kwargs)
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        try:
            manager.set_window_title(title)
        except Exception:
            pass
    return fig, ax
