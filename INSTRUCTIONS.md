# Artifact Instructions

This document is the reproduction guide for the paper’s computationally generated tables and figures. 

- **General Results: DTRL generally produces higher return, task success rate and lower action inconsistency than all other 3 baselines (FULL, SWI and MMS).
Moreover, the effect becomes more pronounced as the task difficulty increases.** (Easiest to hardest: Pendulum < Semicircle-Wide < Semicircle-Narrow < Unitree Go2 Sim.)

## 0. Before Start: Important Notice on Hardware-Dependent Timing Results

Policy inference latency is hardware- and system-dependent even for a same model. Reproducing the paper’s exact numerical values on arbitrary hardware would require hard-coding or emulating both the per-model inference latencies measured on the authors’ workstation and the corresponding time-budget sequence. **But doing so would contradict the purpose of DTRL.**
This artifact employs a real-time dynamic profiling and testing on users' workstation and cpu without any hard-coding to demonstrate the scalability and applicability of our method.
So, the primary reproduction target is the following qualitative behavior claimed in paper:

- **Full Model:** Provides high policy capacity but frequently misses inference deadlines, which can reduce return and task success under dynamic time requirements.
- **SWI:** Generally achieves a high deadline hit rate, but its permanently reduced model capacity can limit task success, especially on more difficult tasks such as Semicircle-Narrow and Unitree Go2.
- **MMS:** Although the deadline hit rate requirement can be met by switching between models, its higher inconsistency leads to a decline in return and task success.
- **DTRL-On and DTRL-Off:** Maintain a strong balance between deadline satisfaction and control performance by selecting among multiple computation levels and low inconsistency at runtime.
- **General Results: DTRL generally produces higher return, task success rate and lower action inconsistency than all other 3 baselines (FULL, SWI and MMS).**

**Artifact Overview and Directory Structure**
You can find more details in [`README.md`](README.md).
The artifact is organized as follows:

```text
 DTRL/
├── checkpoints/ # Training Logs and Checkpoints of all trained models
├── drawing/ # Read data directly from the evaluated results and plot it
├── figure_and_table/ # Save newly generated tables, figures, JSON, and CSV files
├── Quick_Artifact/ # Evaluation and reproduction scripts for all tables and figures
├── environment/ # Dependency specifications
├── third_party/ # Pinned revisions and the FlashSAC patch
├── Src_Training/ # Original training source code for all tasks and methods
├── INSTRUCTIONS.md
├── README.md
└── train_artifact.py
```

Each evaluation-related folder has its own README:

<p class="note-emphasis">The names and purposes of each reproducible computational element, and of the classes, methods, attributes, and variables in the source files, are written in the README.md of the corresponding subfolder.</p>

- [`README.md`](README.md) — repository map and configurable evaluation parameters, one subsection per script
- [`Quick_Artifact/README.md`](Quick_Artifact/README.md) — which eval/plot launcher rebuilds which table or figure
- [`Quick_Artifact/separate/README.md`](Quick_Artifact/separate/README.md) — per-task Table 2 evaluators (Pendulum, both Semicircles, Go2)
- [`drawing/README.md`](drawing/README.md) — plot-only scripts that turn JSON/CSV into PNG
- [`figure_and_table/README.md`](figure_and_table/README.md) — column meanings of the saved JSON/CSV
- [`checkpoints/README.md`](checkpoints/README.md) — layout of the shipped weights that evaluation loads
- [`checkpoints/Ablation/README.md`](checkpoints/Ablation/README.md) — Table 3 and archived Figure 10 ablation weights



## 1. Quick Artifact Evaluation Instruction

Every command in this section is run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
export MPLBACKEND=Agg
```

For latency-sensitive evaluation, avoid running other CPU-intensive programs
at the same time.

### 1.1 Choice 1: Plotting with Precomputed Evaluation Results (1 minute)

This mode uses the evaluation results included in the artifact to regenerate the paper's tables and figures. It does not perform policy evaluation or training.

```bash
# One command: save Tables 2–3 and Figures 6–10, and open the windows
python drawing/plot_all.py
```

```bash
# Headless / save only:
# python drawing/plot_all.py --no-show
# Equivalent step-by-step
# python Quick_Artifact/Table2_MainResults.py --reuse
# python drawing/table2.py --no-show
# python drawing/table3.py --no-show
# python drawing/figure6.py
# python drawing/figure7.py --no-show
# python drawing/figure8.py
# python drawing/figure9.py
# python drawing/figure10.py --no-show
```

**Writes:** `figure_and_table/` (`Table 2 Main Results.csv/.png`,
`Table 3 Ablation2.png`, `figure6.png`, `figure7.png`, `figure8.png`,
`figure9.png`, `figure10.png`).

**Time:** 1 minute.

### 1.2 Choice 2: Re-evaluation and Plotting on the Evaluator's Hardware (5 Hours)

This mode dynamically profiles policy inference latency on the evaluator's CPU, performs the evaluation again, and then regenerates the corresponding tables and figures.
Because inference latency depends on the CPU and system configuration, the numerical results may differ from those reported in the paper. The qualitative trends described in Section 0 are the primary reproduction target.

```bash
# Table 2: four tasks × five methods, then inconsistency (hours)
python Quick_Artifact/Table2_MainResults.py
python drawing/table2.py --no-show

# Table 3: Semicircle-Wide standalone E1/E2/E3 (hours)
python Quick_Artifact/table3_Ablation2.py
python drawing/table3.py --no-show

# Figure 6 from the Table 2 JSON just written
python Quick_Artifact/figure6_prepare.py

# Figure 7: Full without a deadline, then plot vs Table 2 Full (hours)
python Quick_Artifact/figure7_full_withoutdeadline.py

# Figure 8: Semicircle-Narrow DTRL-On E1 eval + plot (needs Table 2 JSON)
python Quick_Artifact/figure8_prepare.py

# Figure 9 from Table 2 inconsistency columns
python Quick_Artifact/figure9_prepard.py

# Figure 10 remains plot-only (shipped TensorBoard logs; no rollout)
python Quick_Artifact/figure10_ablation_warmup.py

# Optional: redraw every PNG from the JSON just written
python drawing/plot_all.py
```

The launchers use the current interpreter for Gym/Safety tasks. Go2 can use a
separate Genesis environment by setting `DTRL_GO2_PY`. To select an explicit
Gym environment as well, set `DTRL_GYM_PY`.

**Writes:** the same `figure_and_table/` files as Choice 1, plus
`figure7_full_compare.json/.csv` and a regenerated Table 3 CSV.

**Time:** many hours (Table 2 is the longest; Figure 7 and Table 3 are next).

---



## 2. Software and Hardware Environment

This artifact is packaged for the evaluation VM below. Choice 1 only redraws
the shipped JSON/CSV; Choice 2 re-evaluates on this machine. Because inference
latency is hardware-dependent (Section 0), Choice 2 numbers need not match the
paper.


| Item                         | Value                                                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Host OS                      | Ubuntu 24.04 LTS, Linux x86_64, kernel 7.0.0-28-generic                                                                                                |
| Shell                        | bash                                                                                                                                                   |
| CPU                          | All evaluation scripts pin to **CPU 0** (`CPU_CORE = 0`); if that core is missing they fall back to the first available core (`os.sched_setaffinity`). |
| Device                       | Evaluation is **CPU**. Optional source training may use CUDA if present.                                                                               |
| Gym / Safety-Gymnasium / SB3 | Python 3.10.19                                                                                                                                        |
| Go2 / Genesis / FlashSAC     | Python 3.13.13                                                                                                                                        |
| Key packages (`sb3sg`)       | `gymnasium==0.28.1`, `numpy==1.23.5`, `torch==2.10.0`, `stable-baselines3==2.7.1`, `mujoco==2.3.3`, `safety-gymnasium==1.2.0`, `matplotlib==3.10.8`    |
| Key packages (`genesis`)     | `genesis==0.4.6`, `torch==2.11.0`, `numpy==2.4.3`                                                                                                      |
| Pins                         | `environment/requirements.txt`, `environment/requirements-go2.txt`, and `environment/environment.yml`                                                 |


Semicircle tasks import `Src_Training/Semicircle-Wide/Semicircle_env` or
`.../Semicircle-Narrow/Semicircle_env` (`v6` = Wide, `v5` = Narrow).

A machine without a `genesis` env can still reproduce every Gym/Safety result (Pendulum, both Semicircles, Table 3, Figures 6/8/9/10, and the non-Go2 parts of Table 2 / Figure 7). Headless plotting: `export MPLBACKEND=Agg`.

Interpreter selection is controlled by the following environment variables:

```
DTRL_GYM_PY=/path/to/gym/python
DTRL_GO2_PY=/path/to/genesis/python
```

Plot scripts (`drawing/*.py`) follow whichever `python` you used to launch them and do not read `GYM_PY` / `GO2_PY`.

## 3. Extra Validated Supported Operating Systems

The original training and evaluation experiments were conducted on Ubuntu 24.04. **The provided artifact was also tested on Ubuntu 22.04, where the documented training, evaluation, and plotting workflows completed successfully.**

Therefore, the provided virtual machine is based on Ubuntu 22.04.

### Python Version Compatibility
The packaged environments use the following tested Python versions:

- **Gym / Safety-Gymnasium / SB3 environment (`sb3sg`):**
  Python 3.10.19. Based on the declared requirements and available package
  builds, Python 3.10.20 and 3.11.15 are also validated.

- **Go2 / Genesis / FlashSAC environment (`genesis`):**
  Python 3.13.13.
  Python 3.12.13 and 3.13.15 are also validated. 



## 4. Optional: Training from Source (Not Required)

As this paper’s evaluation is designed around single-core CPU inference, and the provided VM has only one core, so a full retrain is almost impractical (Even 3 days on more powerful desktop).

This section is only for reviewers who want to inspect or extend training.
There is one **unified launcher** (`train_artifact.py`) and the original
**per-task trainers** under `Src_Training/`. The launcher does not reimplement
any algorithm; it calls those scripts.


Every command is meant to be run from the repository root:

```bash
# source .venv/bin/activate
# export MPLBACKEND=Agg
```

Gym / Safety trainers use `sb3sg`. Go2 trainers use `genesis`.

### 4.1 Unified launcher (`train_artifact.py`)

List targets (does not train):

```bash
# python train_artifact.py --list
```

Train one Gym/Safety target with published defaults. New files go under
`new_trained_checkpoint/<target>/<timestamp>/`:

```bash
# python train_artifact.py --target pendulum-full --mode full \
#   --gym-python "$(command -v python)"
```

Go2 target (Genesis interpreter):

```bash
# python train_artifact.py --target go2-dtrl-off --mode full \
#   --go2-python /path/to/genesis/bin/python
```

All targets sequentially (both interpreters; expected to take days on this VM):

```bash
# python train_artifact.py --all \
#   --gym-python /path/to/gym/bin/python \
#   --go2-python /path/to/genesis/bin/python
```

Arguments after `--` are forwarded to that one trainer:

```bash
# python train_artifact.py --target pendulum-full --mode full -- \
#   --seed 7 --total-timesteps 200000
```

`pendulum-full` also has a tiny `--mode smoke` (64 SAC steps) for a syntax
check. Other targets have no smoke configuration.

| `train_artifact.py --target` | Script | Env |
| --- | --- | --- |
| `pendulum-full` | `Src_Training/Pendulum/Full/train_Full.py` | sb3sg |
| `pendulum-swi` | `Src_Training/Pendulum/SWI/train_SWI.py` | sb3sg |
| `pendulum-mms` | `Src_Training/Pendulum/MMS/train_MMS.py` | sb3sg |
| `pendulum-dtrl-on-e1` / `-e2` | `Src_Training/Pendulum/DTRL-On/train_DTRL_On_e1.py` (and `_e2.py`) | sb3sg |
| `pendulum-dtrl-off` | `Src_Training/Pendulum/DTRL-Off/train_DTRL_Off.py` | sb3sg |
| `semicircle-wide-*` | `Src_Training/Semicircle-Wide/{Full,SWI,MMS,DTRL-On,DTRL-Off}/train_*.py` | sb3sg |
| `semicircle-narrow-*` | `Src_Training/Semicircle-Narrow/{Full,SWI,MMS,DTRL-On,DTRL-Off}/train_*.py` | sb3sg |
| `go2-full` | `Src_Training/Unitree Go2 Sim/Full/full_policy/train_Full.py` | genesis |
| `go2-swi` / `go2-mms` | `Src_Training/Unitree Go2 Sim/SWI/train_SWI.py`, `MMS/train_MMS.py` | genesis |
| `go2-dtrl-on-full` / `-e1` / `-e2` | `Src_Training/Unitree Go2 Sim/DTRL-On/...` | genesis |
| `go2-dtrl-off` | `Src_Training/Unitree Go2 Sim/DTRL-Off/eenn/train_DTRL_Off.py` | genesis |

Trainers that accept an output-root option write under `new_trained_checkpoint/`
directly. Older trainers with a fixed `runs/` directory are scanned afterwards;
new `.zip` / `.pt` / `.pth` files are copied into `collected/`. Each run writes
`run_manifest.json` (command, interpreter, elapsed time, checkpoints).

**Writes:** `new_trained_checkpoint/` (not shipped).

**Time:** impractical on this 1-vCPU VM if run to completion.

### 4.2 Per-task trainers (`Src_Training/`)

Same algorithms as 4.1, invoked directly. `GYM` is the sb3sg interpreter,
`GO2` the genesis interpreter.

```bash
# GYM=/path/to/gym/bin/python
# GO2=/path/to/genesis/bin/python

# --- Pendulum ---
# $GYM Src_Training/Pendulum/Full/train_Full.py
# $GYM Src_Training/Pendulum/SWI/train_SWI.py
# $GYM Src_Training/Pendulum/MMS/train_MMS.py
# $GYM Src_Training/Pendulum/DTRL-On/train_DTRL_On_e1.py
# $GYM Src_Training/Pendulum/DTRL-On/train_DTRL_On_e2.py
# $GYM Src_Training/Pendulum/DTRL-Off/train_DTRL_Off.py

# --- Semicircle-Wide (SafetyPointSemicircle0-v6) ---
# $GYM Src_Training/Semicircle-Wide/Full/train_Full.py
# $GYM Src_Training/Semicircle-Wide/SWI/train_SWI.py
# $GYM Src_Training/Semicircle-Wide/MMS/train_MMS.py
# $GYM Src_Training/Semicircle-Wide/DTRL-On/train_DTRL_On_e1.py
# $GYM Src_Training/Semicircle-Wide/DTRL-On/train_DTRL_On_e2.py
# $GYM Src_Training/Semicircle-Wide/DTRL-On/train_DTRL_On_full.py
# $GYM Src_Training/Semicircle-Wide/DTRL-Off/train_DTRL_Off.py

# --- Semicircle-Narrow (SafetyPointSemicircle0-v5) ---
# $GYM Src_Training/Semicircle-Narrow/Full/train_Full.py
# $GYM Src_Training/Semicircle-Narrow/SWI/train_SWI.py
# $GYM Src_Training/Semicircle-Narrow/MMS/train_MMS.py
# $GYM Src_Training/Semicircle-Narrow/DTRL-On/train_DTRL_On_e1.py
# $GYM Src_Training/Semicircle-Narrow/DTRL-On/train_DTRL_On_e2.py
# $GYM Src_Training/Semicircle-Narrow/DTRL-On/train_DTRL_On_full.py
# $GYM Src_Training/Semicircle-Narrow/DTRL-Off/train_DTRL_Off.py

# --- Unitree Go2 Sim ---
# $GO2 "Src_Training/Unitree Go2 Sim/Full/full_policy/train_Full.py"
# $GO2 "Src_Training/Unitree Go2 Sim/SWI/train_SWI.py"
# $GO2 "Src_Training/Unitree Go2 Sim/MMS/train_MMS.py"
# $GO2 "Src_Training/Unitree Go2 Sim/DTRL-On/full_policy/train_DTRL_On_full.py"
# $GO2 "Src_Training/Unitree Go2 Sim/DTRL-On/train_DTRL_On_e1.py"
# $GO2 "Src_Training/Unitree Go2 Sim/DTRL-On/train_DTRL_On_e2.py"
# $GO2 "Src_Training/Unitree Go2 Sim/DTRL-Off/eenn/train_DTRL_Off.py"
```

Default run directories stay next to each trainer (`.../runs/`). These scripts
are what `train_artifact.py` wraps.

### 4.3 Ablation trainers (Table 3 / Figure 10)

Table 3 is Semicircle-Wide DTRL-Off with two loss mixes (`2:3:5` vs `5:3:2`).
Figure 10 is Pendulum DTRL-Off with vs without full-exit warm-up
(`--warmup-steps`). Reproduction of those figures does **not** need a retrain:
Table 3 loads `checkpoints/Ablation/Ablation2/`, and Figure 10 plots shipped
TensorBoard logs.

```bash
# GYM=/path/to/gym/bin/python

# Table 3 — Semicircle-Wide DTRL-Off, two loss-weight rows
# $GYM Src_Training/Semicircle-Wide/DTRL-Off/train_DTRL_Off.py \
#   --w-exit1 0.2 --w-exit2 0.3 --w-exit3 0.5
# $GYM Src_Training/Semicircle-Wide/DTRL-Off/train_DTRL_Off.py \
#   --w-exit1 0.5 --w-exit2 0.3 --w-exit3 0.2

# Figure 10 — Pendulum DTRL-Off, with vs without efull warm-up
# $GYM Src_Training/Pendulum/DTRL-Off/train_DTRL_Off.py --warmup-steps 28000
# $GYM Src_Training/Pendulum/DTRL-Off/train_DTRL_Off.py --warmup-steps 0
```

### 4.4 How to modify / what will break it

Identified knobs (do not require rewriting the algorithms):

| What | Where |
| --- | --- |
| Number of eval episodes | `N_SEEDS` / `N_EPISODES` in `Quick_Artifact/separate/*.py` and `table3_Ablation2.py` |
| CPU core | `CPU_CORE = 0` in `Quick_Artifact/separate/*.py` (falls back if missing) |
| Interpreter paths | `DTRL_GYM_PY` and `DTRL_GO2_PY` environment variables |
| Table 3 loss mix (if retraining) | `--w-exit1` / `--w-exit2` / `--w-exit3` on `Src_Training/Semicircle-Wide/DTRL-Off/train_DTRL_Off.py` |
| Figure 10 warm-up (if retraining) | `--warmup-steps` on `Src_Training/Pendulum/DTRL-Off/train_DTRL_Off.py` |
| Figure 10 plot smoothing | `WARMUP_STEPS`, `SMOOTHING` in `drawing/figure10.py` |

Likely breakages:

- Pointing `GYM_PY` / `GO2_PY` at `/bin/python3` or an env that lacks Gymnasium / Genesis.
- Running Safety-Gymnasium scripts without `Semicircle_env` on `sys.path` (the artifact scripts add it) or without MuJoCo.
- Running Go2 without `extra_resources/FlashSAC` or without the `genesis` env.
- Running Go2 on a GPU-only Genesis build (this artifact forces CPU).
- Calling plot scripts without `MPLBACKEND=Agg` on a host with no display (`plt.show()` can hang).
- Changing exit network widths in `eenn_network.py` without matching `checkpoints/` (`state_dict` load will fail).
- Uncommenting and running Section 4 on this 1-vCPU VM and expecting it to finish in a review slot.

```
train_artifact.py          optional unified training launcher (this section)
Quick_Artifact/            evaluation commands in Section 1
drawing/                   PNG helpers (also invoked by the prepare scripts)
figure_and_table/          JSON, CSV, PNG outputs
checkpoints/               Table 3 standalone exits and archived Figure 10 weights
Src_Training/              per-task trainers and the run directories Table 2 loads
extra_resources/           vendored Genesis / FlashSAC and conda pins
new_trained_checkpoint/    created only if Section 4 is uncommented (not shipped)
```

---


This artifact does not include real-robot (Unitree Go2 hardware) deployment. Not every laboratory owns a quadruped robot, and the simulation environment already demonstrates the same qualitative trends. In addition, several baseline models can fail easily on hardware and risk damaging the robots.
