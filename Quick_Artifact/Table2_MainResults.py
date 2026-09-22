"""Reproduce Table 2: main results under dynamic inference budgets.

Loads the four task evaluators in ``separate/``, plus
``separate/inconsistence_study.py``. Does not train.

Symbols
-------
Result : one table cell (task × method) with return / hit / success / inconsistency
run : stream a subprocess and return its combined stdout
parse_artifact : pull mean±std lines from a task evaluator
parse_inconsistency : pull task/method inconsistency rows
evaluate : run every task then the inconsistency study
fmt, print_table, main : write JSON/CSV and print the table
"""

import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEPARATE = HERE / "separate"
OUT = HERE.parent / "figure_and_table"
JSON_PATH = OUT / "Table 2 Main Results.json"
CSV_PATH = OUT / "Table 2 Main Results.csv"
GYM_PY = Path(os.environ.get("DTRL_GYM_PY", sys.executable))
GO2_PY = Path(os.environ.get("DTRL_GO2_PY", sys.executable))

METHODS = ("Full", "SWI", "MMS", "DTRL-On", "DTRL-Off")
ALIASES = {
    "FULL": "Full",
    "SWI": "SWI",
    "MMS": "MMS",
    "DTRL-ON": "DTRL-On",
    "DTRL-OFF": "DTRL-Off",
}

TASKS = (
    ("Pendulum", SEPARATE / "Pendulum_artifact.py", GYM_PY),
    ("Semicircle-Wide", SEPARATE / "Semicircle_Wide_artifact.py", GYM_PY),
    ("Semicircle-Narrow", SEPARATE / "Semicircle_Narrow_artifact.py", GYM_PY),
    ("Unitree Go2 Sim", SEPARATE / "Unitree_Go2_artifact.py", GO2_PY),
)

SUMMARY_RE = re.compile(r"^summary \(.+\) — (.+):$")
MEAN_RE = re.compile(
    r"^\s*mean (?:return|hit rate)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*±\s*([-+]?\d+(?:\.\d+)?)$"
)
SUCCESS_RE = re.compile(r"^\s*success rate\s*=\s*([-+]?\d+(?:\.\d+)?)")
INCONSISTENCY_RE = re.compile(
    r"^(Pendulum|Semicircle-Wide|Semicircle-Narrow|Unitree Go2 Sim)\s+"
    r"(MMS|DTRL-On|DTRL-Off)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s*±\s*([-+]?\d+(?:\.\d+)?)\s*$"
)


@dataclass
class Result:
    """One Table 2 row: task name, method name, and the four reported metrics."""

    task: str
    method: str
    return_mean: float
    return_std: float
    hit_mean: float
    hit_std: float
    success_rate: float
    inconsistency_mean: float | None = None
    inconsistency_std: float | None = None


def run(cmd):
    """Run ``cmd``, print stdout live, and return the full text. Raise on nonzero exit."""
    # Stream each task's output so long evaluations remain visible.
    p = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for line in p.stdout:
        print(line, end="")
        lines.append(line)
    if p.wait() != 0:
        raise RuntimeError("".join(lines))
    return "".join(lines)


def parse_artifact(task, output):
    """Parse ``summary (...) — Method:`` blocks from a ``*_artifact.py`` log into Result rows."""
    parsed = {}
    current = None
    for line in output.splitlines():
        m = SUMMARY_RE.match(line.strip())
        if m:
            current = ALIASES.get(m.group(1).strip().upper())
            if current:
                parsed[current] = {}
            continue
        if current is None:
            continue
        if line.lstrip().startswith("mean return"):
            m = MEAN_RE.match(line)
            if m:
                parsed[current]["return_mean"] = float(m.group(1))
                parsed[current]["return_std"] = float(m.group(2))
        elif line.lstrip().startswith("mean hit rate"):
            m = MEAN_RE.match(line)
            if m:
                parsed[current]["hit_mean"] = float(m.group(1))
                parsed[current]["hit_std"] = float(m.group(2))
        elif line.lstrip().startswith("success rate"):
            m = SUCCESS_RE.match(line)
            if m:
                parsed[current]["success_rate"] = float(m.group(1))
    return [Result(task=task, method=method, **parsed[method]) for method in METHODS]


def parse_inconsistency(output):
    """Map (task, method) to (mean, std) from the inconsistency-study table."""
    out = {}
    for line in output.splitlines():
        m = INCONSISTENCY_RE.match(line.strip())
        if m:
            out[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
    return out


def evaluate():
    """Evaluate all four tasks, then attach inconsistency numbers where they exist."""
    rows = []
    # Go2 uses its Genesis environment; the other tasks share the Gym setup.
    for name, script, python in TASKS:
        text = run([str(python), str(script)])
        rows.extend(parse_artifact(name, text))
    text = run([sys.executable, str(SEPARATE / "inconsistence_study.py")])
    inc = parse_inconsistency(text)
    for row in rows:
        if (row.task, row.method) in inc:
            row.inconsistency_mean, row.inconsistency_std = inc[(row.task, row.method)]
    return rows


def fmt(mean, std):
    """Format a mean±std cell with four decimal places."""
    return f"{mean:.4f} ± {std:.4f}"


def print_table(rows):
    """Print the six-column Table 2 text view."""
    print()
    header = (
        f"{'Task':22s} {'Method':10s} {'Return':23s} "
        f"{'Deadline Hit Rate':23s} {'Success':8s} {'Inconsistency':23s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        inc = (
            fmt(r.inconsistency_mean, r.inconsistency_std)
            if r.inconsistency_mean is not None
            else "–"
        )
        print(
            f"{r.task:22s} {r.method:10s} "
            f"{fmt(r.return_mean, r.return_std):23s} "
            f"{fmt(r.hit_mean, r.hit_std):23s} "
            f"{100.0 * r.success_rate:7.0f}% {inc:23s}"
        )


def main():
    """Evaluate (or ``--reuse`` JSON) and write ``Table 2 Main Results.json/.csv``."""
    reuse = "--reuse" in sys.argv
    if reuse:
        rows = [Result(**x) for x in json.loads(JSON_PATH.read_text())]
    else:
        rows = evaluate()
        OUT.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps([asdict(r) for r in rows], indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Task", "Method", "Return", "Deadline Hit Rate", "Success Rate", "Inconsistency"])
        for r in rows:
            inc = (
                fmt(r.inconsistency_mean, r.inconsistency_std)
                if r.inconsistency_mean is not None
                else "–"
            )
            w.writerow(
                [
                    r.task,
                    r.method,
                    fmt(r.return_mean, r.return_std),
                    fmt(r.hit_mean, r.hit_std),
                    f"{100.0 * r.success_rate:.0f}%",
                    inc,
                ]
            )
    print_table(rows)
    print(f"\nJSON:  {JSON_PATH}")
    print(f"CSV:   {CSV_PATH}")


if __name__ == "__main__":
    main()
