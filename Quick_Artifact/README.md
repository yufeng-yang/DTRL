# Quick_Artifact

Scripts that **load trained weights and evaluate** (or plot already-saved
numbers). None of these files train the Table 2 main-result policies.

## Files

| File | Purpose |
|---|---|
| `Table2_MainResults.py` | Run four task evaluators + inconsistency study; write Table 2 JSON/CSV |
| `table3_Ablation2.py` | Evaluate standalone E1/E2/E3 weights for two loss-weight rows |
| `figure6_prepare.py` | Plot Figure 6 from Table 2 JSON |
| `figure7_full_withoutdeadline.py` | Re-evaluate Full **without** a deadline, then plot vs Table 2 Full |
| `figure8_prepare.py` | Semicircle-Narrow: SWI vs DTRL-On E1-only vs dynamic DTRL-On |
| `figure9_prepard.py` | Copy inconsistency columns from Table 2 JSON into Figure 9 |
| `figure10_ablation_warmup.py` | Read the shipped TensorBoard logs and plot the Figure 10 warm-up curves; no training |
| `separate/Pendulum_artifact.py` | Table 2 evaluator for `Pendulum-v1` (CPU 0) |
| `separate/Semicircle_Wide_artifact.py` | Table 2 evaluator for `SafetyPointSemicircle0-v6` (CPU 0) |
| `separate/Semicircle_Narrow_artifact.py` | Table 2 evaluator for `SafetyPointSemicircle0-v5` (CPU 0) |
| `separate/Unitree_Go2_artifact.py` | Table 2 evaluator for Go2 Genesis (CPU 0) |
| `separate/inconsistence_study.py` | Cross-exit action inconsistency for MMS / DTRL-On / DTRL-Off |

The launchers use the current Python interpreter by default. Optional
environment variables select separate environments:

- `DTRL_GYM_PY` — Gym / Safety Gymnasium
- `DTRL_GO2_PY` — Genesis / Unitree Go2

Checkpoint paths inside `Src_Training/*/evaluation/` still mention the original
machine. `Table2_MainResults.py` / `figure7_full_withoutdeadline.py` /
`separate/inconsistence_study.py` rewrite them onto this repository before load.
