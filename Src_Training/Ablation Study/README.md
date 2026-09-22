# Ablation Study

| Folder | Paper | What it changes |
|---|---|---|
| `Full-Exit Warm-up/` | Figure 10 | Pendulum DTRL-Off, with vs without training only efull first |
| `Loss Weight Allocation/` | Table 3 | Semicircle-Wide DTRL-Off, loss mix `0.2/0.3/0.5` vs `0.5/0.3/0.2` |

The shipped TensorBoard logs under `Full-Exit Warm-up/runs/` are the data source
for `Quick_Artifact/figure10_ablation_warmup.py`; that entry point only plots
the recorded curves and performs no training.
`Loss Weight Allocation/run_experiment.py` launches the original Wide
`train_DTRL_Off.py` twice with different `--w-exit*` flags.
`Loss Weight Allocation/pick_table_point.py` reads TensorBoard to pick
eval steps; `reeval_ckpts.py` / `reeval_semicircle.py` re-score saved
checkpoints. None of these files modify the original trainers.
