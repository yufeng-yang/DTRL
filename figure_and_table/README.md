# figure_and_table

Numeric outputs and rendered figures. JSON/CSV are UTF-8. Every CSV has a
header row. JSON uses the same field names as the CSV unless noted.

`return_*` is undiscounted episode return. `hit_*` is the fraction of
control steps whose inference finished inside the sampled deadline.
`success_rate` is in `[0, 1]` (not percent). `inconsistency_*` is the
mean/std of the cross-exit action-disagreement metric from the paper.

## Table 2 Main Results.csv / .json

One row per (task, method).

| Field | Meaning |
|---|---|
| `task` | `Pendulum`, `Semicircle-Wide`, `Semicircle-Narrow`, or `Unitree Go2 Sim` |
| `method` | `Full`, `SWI`, `MMS`, `DTRL-On`, or `DTRL-Off` |
| `return_mean`, `return_std` | Episode return over evaluation seeds |
| `hit_mean`, `hit_std` | Deadline hit rate over those episodes |
| `success_rate` | Task success fraction |
| `inconsistency_mean`, `inconsistency_std` | Filled for MMS/DTRL-On/DTRL-Off; empty/null for Full and SWI |

## Table 3 Ablation2.csv / .json

Semicircle-Wide DTRL-Off, two loss-weight rows, three exits.

| Field | Meaning |
|---|---|
| `Loss Weights (E1 : E2 : E3)` / `loss_weights` | Training loss mix, e.g. `2 : 3 : 5` |
| `E1` / `e1` | Mean return of Exit 1 (`obs→256→out`) |
| `E2` / `e2` | Mean return of Exit 2 (`obs→256→128→128→out`) |
| `E3` / `e3` | Mean return of Exit 3 / efull |
| `n_episodes` | Episodes used for that JSON dump (CSV is the rounded mean) |
| `folder` | Weight directory under `checkpoints/Ablation/Ablation2/` |

## figure7_full_compare.csv / .json

Full policy evaluated **without** a deadline (deadline set to a huge value).

| Field | Meaning |
|---|---|
| `task`, `short_name` | Full task name and plot label |
| `success_rate`, `return_mean`, `return_std` | Same definitions as Table 2 |

Figure 7’s “with deadline” bars come from Table 2 Full rows.

## figure8.csv / .json

Semicircle-Narrow only.

| Field | Meaning |
|---|---|
| `task` | Always `Semicircle-Narrow` |
| `method` | `SWI`, `DTRL-On E1`, or `DTRL-On` |
| `success_rate`, `return_*`, `hit_*` | Same as Table 2 |
| `n` | Number of evaluation episodes |
| `budget_ms` | JSON-only: `[low, high]` measured inference budget in milliseconds |

## figure9.csv / .json

| Field | Meaning |
|---|---|
| `task`, `short_name` | Full name and plot label |
| `method` | `MMS`, `DTRL-On`, or `DTRL-Off` |
| `inconsistency_mean`, `inconsistency_std` | Copied from Table 2 |

JSON stores one object per method, each with a list aligned to `tasks`.

## PNG files

`figure6.png` … `figure10.png`, `Table 2 Main Results.png`, and
`Table 3 Ablation2.png` are the plots drawn from the JSON/CSV in this folder.
