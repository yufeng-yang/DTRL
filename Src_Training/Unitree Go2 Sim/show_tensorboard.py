"""Start TensorBoard to view the event file written by rsl-rl training.

TensorBoard's --logdir must point to the "folder containing events files" and cannot point to
events.out.tfevents.* files themselves. Please write the directory of a certain run (such as full_timestamp) into
Run after TENSORBOARD_LOGDIR:
  python show_tensorboard.py

Dependencies: pip install tensorboard"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# The run root directory of this full_train (there should be events.out.tfevents.* under it, do not write the file name into the path)
TENSORBOARD_LOGDIR = Path(
    "/home/yufeng.yang/codespace/Prepare_to_transfer/src_training/Unitree Go2 Sim/onpolicy/runs/onpolicy/full_20260424_103256_091127"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start TensorBoard on TENSORBOARD_LOGDIR")
    parser.add_argument("--port", type=int, default=6006, help="HTTP port, default 6006")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Binding address; available when other machines on the LAN need to access 0.0.0.0",
    )
    args = parser.parse_args()

    root = TENSORBOARD_LOGDIR.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(
            f"Not a valid directory: {root}\nPlease modify TENSORBOARD_LOGDIR in this file to the actual training output path."
        )

    tb = shutil.which("tensorboard")
    if tb is None:
        raise SystemExit(
            "tensorboard command not found. Please execute first: pip install tensorboard"
        )

    cmd = [tb, f"--logdir={root}", f"--port={args.port}", f"--host={args.host}"]
    print("logdir =", root)
    print("start up:", " ".join(cmd))
    show_host = "127.0.0.1" if args.host in ("0.0.0.0", "[::]") else args.host
    print(f"Browser: http://{show_host}:{args.port}/")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nExited", file=sys.stderr)
        raise SystemExit(0) from None
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.returncode) from e


if __name__ == "__main__":
    main()
