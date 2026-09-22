#!/usr/bin/env python3
"""Unified launcher for the training programs under ``Src_Training``.

The launcher does not reimplement any algorithm.  It selects an existing
trainer, gives it a run directory under ``new_trained_checkpoint/`` when the
trainer supports that option, and collects newly created checkpoint files for
older trainers whose output path is fixed internally.

Run ``python train_artifact.py --list`` to see the available targets.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "new_trained_checkpoint"
CHECKPOINT_SUFFIXES = {".zip", ".pt", ".pth"}


@dataclass(frozen=True)
class Target:
    """One public name mapped to one existing training entry point."""

    name: str
    script: str
    environment: str
    output_option: str | None = None
    smoke_args: tuple[str, ...] | None = None
    verifier: str = "auto"
    legacy_scan_root: str | None = None

    @property
    def script_path(self) -> Path:
        return ROOT / self.script

    @property
    def scan_root(self) -> Path:
        return ROOT / self.legacy_scan_root if self.legacy_scan_root else self.script_path.parent


PENDULUM_SMOKE_ARGS = (
    "--total-timesteps",
    "64",
    "--n-envs",
    "1",
    "--learning-starts",
    "8",
    "--batch-size",
    "8",
    "--buffer-size",
    "128",
    "--save-freq",
    "32",
    "--eval-freq",
    "64",
    "--n-eval-seeds",
    "1",
    "--verbose",
    "0",
)


TARGETS = {
    target.name: target
    for target in (
        Target(
            "pendulum-full",
            "Src_Training/Pendulum/Full/train_Full.py",
            "gym",
            "--root",
            PENDULUM_SMOKE_ARGS,
            "sb3-sac",
        ),
        Target("pendulum-swi", "Src_Training/Pendulum/SWI/train_SWI.py", "gym", "--root", verifier="sb3-sac"),
        Target("pendulum-mms", "Src_Training/Pendulum/MMS/train_MMS.py", "gym", "--root", verifier="sb3-sac"),
        Target("pendulum-dtrl-on-e1", "Src_Training/Pendulum/DTRL-On/train_DTRL_On_e1.py", "gym", verifier="sb3-ppo"),
        Target("pendulum-dtrl-on-e2", "Src_Training/Pendulum/DTRL-On/train_DTRL_On_e2.py", "gym", verifier="sb3-ppo"),
        Target("pendulum-dtrl-off", "Src_Training/Pendulum/DTRL-Off/train_DTRL_Off.py", "gym", "--log-root", verifier="torch"),
        Target("semicircle-wide-full", "Src_Training/Semicircle-Wide/Full/train_Full.py", "gym", "--runs-root", verifier="sb3-sac"),
        Target("semicircle-wide-swi", "Src_Training/Semicircle-Wide/SWI/train_SWI.py", "gym", "--runs-root", verifier="sb3-sac"),
        Target("semicircle-wide-mms", "Src_Training/Semicircle-Wide/MMS/train_MMS.py", "gym", "--runs-root", verifier="sb3-ppo"),
        Target("semicircle-wide-dtrl-on-full", "Src_Training/Semicircle-Wide/DTRL-On/train_DTRL_On_full.py", "gym", "--runs-root", verifier="sb3-ppo"),
        Target("semicircle-wide-dtrl-on-e1", "Src_Training/Semicircle-Wide/DTRL-On/train_DTRL_On_e1.py", "gym", "--runs-root", verifier="sb3-ppo"),
        Target("semicircle-wide-dtrl-on-e2", "Src_Training/Semicircle-Wide/DTRL-On/train_DTRL_On_e2.py", "gym", "--runs-root", verifier="sb3-ppo"),
        Target("semicircle-wide-dtrl-off", "Src_Training/Semicircle-Wide/DTRL-Off/train_DTRL_Off.py", "gym", "--runs-root", verifier="torch"),
        Target("semicircle-narrow-full", "Src_Training/Semicircle-Narrow/Full/train_Full.py", "gym", "--runs-root", verifier="sb3-sac"),
        Target("semicircle-narrow-swi", "Src_Training/Semicircle-Narrow/SWI/train_SWI.py", "gym", "--runs-root", verifier="sb3-ppo"),
        Target("semicircle-narrow-mms", "Src_Training/Semicircle-Narrow/MMS/train_MMS.py", "gym", "--runs-root", verifier="sb3-ppo"),
        Target("semicircle-narrow-dtrl-on-full", "Src_Training/Semicircle-Narrow/DTRL-On/train_DTRL_On_full.py", "gym", "--runs-root", verifier="sb3-ppo"),
        Target("semicircle-narrow-dtrl-on-e1", "Src_Training/Semicircle-Narrow/DTRL-On/train_DTRL_On_e1.py", "gym", "--runs-root", verifier="sb3-ppo"),
        Target("semicircle-narrow-dtrl-on-e2", "Src_Training/Semicircle-Narrow/DTRL-On/train_DTRL_On_e2.py", "gym", "--runs-root", verifier="sb3-ppo"),
        Target("semicircle-narrow-dtrl-off", "Src_Training/Semicircle-Narrow/DTRL-Off/train_DTRL_Off.py", "gym", "--runs-root", verifier="torch"),
        Target(
            "go2-full",
            "Src_Training/Unitree Go2 Sim/Full/full_policy/train_Full.py",
            "go2",
            verifier="torch",
            legacy_scan_root="Src_Training/Unitree Go2 Sim/Full",
        ),
        Target("go2-swi", "Src_Training/Unitree Go2 Sim/SWI/train_SWI.py", "go2", verifier="torch"),
        Target("go2-mms", "Src_Training/Unitree Go2 Sim/MMS/train_MMS.py", "go2", verifier="torch"),
        Target(
            "go2-dtrl-on-full",
            "Src_Training/Unitree Go2 Sim/DTRL-On/full_policy/train_DTRL_On_full.py",
            "go2",
            verifier="torch",
            legacy_scan_root="Src_Training/Unitree Go2 Sim/DTRL-On",
        ),
        Target("go2-dtrl-on-e1", "Src_Training/Unitree Go2 Sim/DTRL-On/train_DTRL_On_e1.py", "go2", verifier="torch"),
        Target("go2-dtrl-on-e2", "Src_Training/Unitree Go2 Sim/DTRL-On/train_DTRL_On_e2.py", "go2", verifier="torch"),
        Target("go2-dtrl-off", "Src_Training/Unitree Go2 Sim/DTRL-Off/eenn/train_DTRL_Off.py", "go2", "--runs-root", verifier="torch"),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run existing Src_Training trainers and centralize their new checkpoints."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--target", action="append", choices=sorted(TARGETS), help="Training target; repeat to run several.")
    selection.add_argument("--all", action="store_true", help="Run every target sequentially with its published defaults.")
    parser.add_argument("--list", action="store_true", help="List targets and exit without training.")
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gym-python", type=Path, help="Python from the sb3sg environment; defaults to this interpreter.")
    parser.add_argument("--go2-python", type=Path, help="Python from the Genesis environment; required for Go2 targets when different.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running trainers or creating outputs.")
    parser.add_argument("--keep-going", action="store_true", help="With multiple targets, continue after a failed target.")
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments for one target, placed after '--'.",
    )
    return parser.parse_args()


def list_targets() -> None:
    print("Available training targets:")
    for name, target in TARGETS.items():
        smoke = "yes" if target.smoke_args is not None else "no"
        print(f"  {name:34s} env={target.environment:4s} smoke={smoke}  {target.script}")


def snapshot_checkpoints(root: Path) -> dict[Path, tuple[int, int]]:
    """Return size and nanosecond mtime for checkpoints under ``root``."""

    if not root.exists():
        return {}
    result: dict[Path, tuple[int, int]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES:
            stat = path.stat()
            result[path.resolve()] = (stat.st_size, stat.st_mtime_ns)
    return result


def changed_checkpoints(before: dict[Path, tuple[int, int]], root: Path) -> list[Path]:
    after = snapshot_checkpoints(root)
    return sorted(path for path, state in after.items() if before.get(path) != state)


def collect_legacy_outputs(paths: Iterable[Path], source_root: Path, destination: Path) -> list[Path]:
    collected: list[Path] = []
    for source in paths:
        try:
            relative = source.relative_to(source_root.resolve())
        except ValueError:
            relative = Path(source.name)
        target = destination / "collected" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        collected.append(target)
    return collected


def verify_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"not a valid ZIP checkpoint: {path}")
    with zipfile.ZipFile(path) as archive:
        broken = archive.testzip()
    if broken is not None:
        raise RuntimeError(f"corrupt member {broken!r} in {path}")


def verify_with_interpreter(path: Path, verifier: str, interpreter: Path, env: dict[str, str]) -> None:
    if verifier in {"sb3-sac", "sb3-ppo"}:
        verify_zip(path)
        algorithm = "SAC" if verifier == "sb3-sac" else "PPO"
        code = (
            f"from stable_baselines3 import {algorithm}; "
            f"{algorithm}.load({str(path)!r}, device='cpu'); print('load ok')"
        )
    elif path.suffix.lower() in {".pt", ".pth"} or verifier == "torch":
        code = (
            "import torch; "
            f"torch.load({str(path)!r}, map_location='cpu', weights_only=False); print('load ok')"
        )
    else:
        verify_zip(path)
        return
    subprocess.run([str(interpreter), "-c", code], check=True, cwd=ROOT, env=env)


def run_target(target: Target, args: argparse.Namespace) -> Path:
    if args.mode == "smoke" and target.smoke_args is None:
        raise RuntimeError(f"{target.name} has no bounded smoke configuration; use --mode full")

    interpreter = args.gym_python if target.environment == "gym" else args.go2_python
    interpreter = (interpreter or Path(sys.executable)).expanduser().resolve()
    if not interpreter.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {interpreter}")
    if not target.script_path.is_file():
        raise FileNotFoundError(f"Training script not found: {target.script_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root.resolve() / target.name / stamp
    command = [str(interpreter), str(target.script_path)]
    if target.output_option is not None:
        command.extend([target.output_option, str(run_dir)])
    if args.mode == "smoke":
        command.extend(target.smoke_args or ())
    command.extend(args.trainer_args)

    print(f"\n[{target.name}] {' '.join(command)}")
    if args.dry_run:
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = run_dir / ".cache"
    cache_dir.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
            "NUMBA_CACHE_DIR": str(cache_dir / "numba"),
            "PYTHONUNBUFFERED": "1",
        }
    )

    legacy_root = target.scan_root
    before = snapshot_checkpoints(legacy_root) if target.output_option is None else {}
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    completed = subprocess.run(command, cwd=target.script_path.parent, env=environment)
    elapsed = time.monotonic() - started

    if completed.returncode != 0:
        status = "failed"
        checkpoints: list[Path] = []
    elif target.output_option is None:
        checkpoints = collect_legacy_outputs(
            changed_checkpoints(before, legacy_root), legacy_root, run_dir
        )
        status = "completed"
    else:
        checkpoints = sorted(
            path for path in run_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES
        )
        status = "completed"

    verified: list[str] = []
    verification_errors: list[str] = []
    if status == "completed":
        if not checkpoints:
            verification_errors.append("trainer completed but no .zip/.pt/.pth checkpoint was found")
        for checkpoint in checkpoints:
            try:
                verify_with_interpreter(checkpoint, target.verifier, interpreter, environment)
                verified.append(str(checkpoint.relative_to(run_dir)))
            except Exception as exc:  # preserve every run's diagnostic manifest
                verification_errors.append(f"{checkpoint}: {exc}")

    manifest = {
        "target": asdict(target),
        "mode": args.mode,
        "status": status,
        "command": command,
        "interpreter": str(interpreter),
        "started_utc": started_utc,
        "elapsed_seconds": round(elapsed, 3),
        "checkpoints": [str(path.relative_to(run_dir)) for path in checkpoints],
        "verified_checkpoints": verified,
        "verification_errors": verification_errors,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    if verification_errors:
        raise RuntimeError("; ".join(verification_errors))
    print(f"[{target.name}] verified {len(verified)} checkpoint(s) under {run_dir}")
    return run_dir


def main() -> int:
    args = parse_args()
    if args.trainer_args[:1] == ["--"]:
        args.trainer_args = args.trainer_args[1:]
    if args.list:
        list_targets()
        return 0

    names = list(TARGETS) if args.all else (args.target or [])
    if not names:
        raise SystemExit("choose --target NAME, --all, or --list")
    if args.trainer_args and len(names) != 1:
        raise SystemExit("extra trainer arguments after '--' require exactly one --target")
    if args.mode == "smoke" and args.all:
        raise SystemExit("smoke mode is intentionally bounded; select a smoke-capable target")

    failures: list[str] = []
    for name in names:
        try:
            run_target(TARGETS[name], args)
        except Exception as exc:
            print(f"[{name}] ERROR: {exc}", file=sys.stderr)
            failures.append(name)
            if not args.keep_going:
                break
    if failures:
        print(f"Failed targets: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
