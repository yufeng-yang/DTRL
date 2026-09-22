# checkpoints

Trained weights consumed by `Quick_Artifact/`. Each method folder has a
`SOURCES.md` that names the training run the files were copied from.

## Layout

```
checkpoints/
  Pendulum|Semicircle-Wide|Semicircle-Narrow|Unitree Go2 Sim/
    Full|SWI|MMS|DTRL-On|DTRL-Off/
      SOURCES.md
      *.zip or *.pt          # SB3 zip or joint SAC checkpoint
      split/                 # optional standalone DTRL-Off exits
  Ablation/
    Ablation1/               # Archived Figure 10 Pendulum warm-up weights (plot uses TensorBoard logs)
    Ablation2/2-3-5|5-3-2/   # Table 3 standalone E1/E2/E3
```

## `.pt` dictionaries (DTRL-Off / Table 3)

Joint SAC checkpoints typically contain:

| Key | Meaning |
|---|---|
| `actor` | Joint three-exit actor `state_dict` |
| `critic`, `critic_target` | Twin Q networks |
| `opt_a`, `opt_c`, `log_alpha` | Optimizer / temperature, when saved |

Standalone Table 3 exits (`Ablation/Ablation2/.../e1.pt` etc.) contain:

| Key | Meaning |
|---|---|
| `tag`, `exit_id` | `e1`/`e2`/`e3` and integer 1/2/3 |
| `arch` | Layer string, e.g. `obs→256→out` |
| `env_id` | Gym id, `SafetyPointSemicircle0-v6` for Table 3 |
| `obs_dim`, `act_dim` | Network I/O sizes |
| `table_row` | `2:3:5` or `5:3:2` |
| `source_run`, `source_step`, `source_exit` | Which joint run/step/exit was exported |
| `eval_episodes`, `eval_seeds`, `mean_return` | How the saved return was measured |
| `joint_model` | Path of the joint checkpoint |
| `state_dict` | Weights for that exit only (`backbone`, extra MLP blocks, `mean`) |

SB3 `*.zip` files are Stable-Baselines3 PPO/SAC saves (`PPO.load` / `SAC.load`).
