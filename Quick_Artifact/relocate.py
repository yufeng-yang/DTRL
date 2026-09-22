"""Map leftover absolute paths from the original training machine onto this repo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_OLD_PREFIXES = (
    (
        "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training",
        "Src_Training",
    ),
    (
        "/home/yufeng.yang/codespace/Prepare_to_transfer/Src_Training",
        "Src_Training",
    ),
    (
        "/home/yufeng.yang/codespace/RTSS/FlashSAC",
        "extra_resources/FlashSAC",
    ),
    (
        "/home/yufeng.yang/codespace/Prepare_to_transfer",
        "",
    ),
)


def relocate_str(text: str, repo: Path) -> str:
    """Rewrite one path string onto ``repo`` when it still points at the old host."""
    for old, rel in _OLD_PREFIXES:
        if text == old or text.startswith(old + "/"):
            mapped = repo if rel == "" else repo / rel
            return str(mapped) + text[len(old) :]
    if "/src_training/" in text:
        return text.replace("/src_training/", "/Src_Training/")
    return text


def relocate(obj: Any, repo: Path) -> Any:
    """Recursively rewrite Path / str / list / tuple / dict values."""
    if isinstance(obj, Path):
        return Path(relocate_str(str(obj), repo))
    if isinstance(obj, str):
        return relocate_str(obj, repo)
    if isinstance(obj, tuple):
        return tuple(relocate(x, repo) for x in obj)
    if isinstance(obj, list):
        return [relocate(x, repo) for x in obj]
    if isinstance(obj, dict):
        return {relocate(k, repo): relocate(v, repo) for k, v in obj.items()}
    return obj


def _needs_rewrite(obj: Any) -> bool:
    if isinstance(obj, Path):
        return True
    if isinstance(obj, str):
        return obj.startswith("/home/yufeng.yang/") or "/src_training/" in obj
    if isinstance(obj, (list, tuple)):
        return any(_needs_rewrite(x) for x in obj)
    if isinstance(obj, dict):
        return any(_needs_rewrite(k) or _needs_rewrite(v) for k, v in obj.items())
    return False


def relocate_module(mod: Any, repo: Path) -> None:
    """Patch path-bearing globals on a loaded Src_Training evaluation module."""
    for name, value in list(vars(mod).items()):
        if name.startswith("__") or not _needs_rewrite(value):
            continue
        setattr(mod, name, relocate(value, repo))
