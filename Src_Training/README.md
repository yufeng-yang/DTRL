# Src_Training

Training code and older per-task evaluators. Table 2 reproduction should
start from `Quick_Artifact/`, which already points at the runs listed here.

## Tasks

Each of `Pendulum/`, `Semicircle-Wide/`, `Semicircle-Narrow/`, and
`Unitree Go2 Sim/` contains one trainer per method:

| Path pattern | Purpose |
|---|---|
| `Full/train_Full.py` | Unconstrained full-depth baseline |
| `SWI/train_SWI.py` | Anytime / switching baseline |
| `MMS/train_MMS.py` | Multi-model switching baseline |
| `DTRL-On/train_DTRL_On_e1.py` | On-policy DTRL Exit 1 |
| `DTRL-On/train_DTRL_On_e2.py` | On-policy DTRL Exit 2 |
| `DTRL-On/train_DTRL_On_full.py` | On-policy DTRL full / Exit 3 (where present) |
| `DTRL-Off/train_DTRL_Off.py` | Off-policy joint SAC with three exits |
| `DTRL-Off/eenn_network.py` | Actor/critic layouts for that task |
| `DTRL-Off/action_utils.py` | Scale/unscale actions to the env box |
| `evaluation/` | Standalone eval, timing, render, inconsistency helpers |
| `env_show.py` / `render.py` / `datashow.py` | Visualization only; not used for tables |

Pendulum DTRL-On has `train_DTRL_On_e1.py` and `train_DTRL_On_e2.py` plus a
full trainer under the same folder naming used by the other tasks.

Semicircle maps: `SafetyPointSemicircle0-v6` is Wide, `v5` is Narrow.
Registration code lives in `Semicircle_env/` (see that folder’s README).

## Ablation Study

| Path | Purpose |
|---|---|
| `Ablation Study/Full-Exit Warm-up/` | Figure 10: Pendulum DTRL-Off with vs without efull warm-up |
| `Ablation Study/Loss Weight Allocation/` | Table 3: Semicircle-Wide 2:3:5 vs 5:3:2 loss mix |

The Figure 10 entry point no longer launches an ablation trainer. It reads the
shipped TensorBoard event files under `Ablation Study/Full-Exit Warm-up/runs/`
and plots their recorded Exit 1/2/3 evaluation returns.
