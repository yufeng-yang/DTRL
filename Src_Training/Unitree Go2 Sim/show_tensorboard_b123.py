"""Compare the training curves of b1_full / b2_owf / b3_mms in the same TensorBoard.

Log directory (rsl-rl writes events.out.tfevents.* in each run root directory by default)::

    b1_full/full_policy/full_<timestamp>/
    b2_owf/runs/swi_<timestamp>/
    b3_mms/runs/mms2_<timestamp>/

Run::

    python unitree_go2/show_tensorboard_b123.py
    python unitree_go2/show_tensorboard_b123.py --port 6007

Dependencies: pip install tensorboard"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_COMPARE_ROOT = _ROOT / ".tb_compare_b123"

# Each module searches the root directory and run name prefix
_SEARCH: list[tuple[str, Path, str]] = [
    ("Full", _ROOT / "Full" / "full_policy", "Full_"),
    ("SWI", _ROOT / "SWI" / "runs", "SWI_"),
    ("MMS", _ROOT / "MMS" / "runs", "MMS_"),
]


def _has_tfevents(d: Path) -> bool:
    return d.is_dir() and any(d.glob("events.out.tfevents*"))


def _latest_run(parent: Path, prefix: str) -> Path | None:
    if not parent.is_dir():
        return None
    candidates = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    candidates = [p for p in candidates if _has_tfevents(p)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _discover_runs(explicit: dict[str, str | None]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for label, parent, prefix in _SEARCH:
        if explicit.get(label):
            p = Path(explicit[label]).expanduser().resolve()
            if not _has_tfevents(p):
                raise SystemExit(f"[{label}] There is no events file in the directory: {p}")
            found[label] = p
            continue
        run = _latest_run(parent, prefix)
        if run is None:
            raise SystemExit(f"[{label}] Not here {parent} Find the one containing events {prefix}* run")
        found[label] = run
    return found


def _prepare_compare_dir(runs: dict[str, Path]) -> Path:
    if _COMPARE_ROOT.exists():
        shutil.rmtree(_COMPARE_ROOT)
    _COMPARE_ROOT.mkdir(parents=True)
    for label, src in runs.items():
        link = _COMPARE_ROOT / label
        link.symlink_to(src.resolve())
    return _COMPARE_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="b1_full / b2_owf / b3_mms combined with TensorBoard")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--b1", type=str, default=None, help="b1_full run directory (including events)")
    parser.add_argument("--b2", type=str, default=None, help="b2_owf run directory")
    parser.add_argument("--b3", type=str, default=None, help="b3_mms run directory")
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Instead of using symlink to aggregate directories, use --logdir_spec (not supported by some older versions of TB)",
    )
    args = parser.parse_args()

    runs = _discover_runs({"b1_full": args.b1, "b2_owf": args.b2, "b3_mms": args.b3})

    tb = shutil.which("tensorboard")
    if tb is None:
        raise SystemExit("tensorboard not found, please execute: pip install tensorboard")

    print("The following run will be loaded (left RUN optional b1_full / b2_owf / b3_mms):")
    for label, p in runs.items():
        print(f"  {label}: {p}")

    if args.no_symlink:
        spec = ",".join(f"{k}:{v}" for k, v in runs.items())
        cmd = [tb, f"--logdir_spec={spec}", f"--port={args.port}", f"--host={args.host}"]
    else:
        logdir = _prepare_compare_dir(runs)
        cmd = [tb, f"--logdir={logdir}", f"--port={args.port}", f"--host={args.host}"]

    show_host = "127.0.0.1" if args.host in ("0.0.0.0", "[::]") else args.host
    print(f"\nStart: {' '.join(cmd)}")
    print(f"Browser: http://{show_host}:{args.port}/")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
