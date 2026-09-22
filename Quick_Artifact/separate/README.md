# Quick_Artifact/separate

Per-task evaluators called by `Table2_MainResults.py`. Each file loads
that task’s trained Full / SWI / MMS / DTRL-On / DTRL-Off weights,
measures return, deadline hit rate, and success, and prints a `summary`
block the launcher parses.

| File | Environment | CPU core |
|---|---|---|
| `Pendulum_artifact.py` | `Pendulum-v1` | 0 |
| `Semicircle_Wide_artifact.py` | `SafetyPointSemicircle0-v6` | 0 |
| `Semicircle_Narrow_artifact.py` | `SafetyPointSemicircle0-v5` | 0 |
| `Unitree_Go2_artifact.py` | Genesis Go2 | 0 |
| `inconsistence_study.py` | all four (inconsistency only) | inherits worker env |
