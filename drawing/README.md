# drawing

Plot-only scripts. They read JSON/CSV under `figure_and_table/` and write
PNG files next to those data files. They do not load neural-network weights.

| File | Purpose |
|---|---|
| `figure6.py` | Deadline hit rate vs success rate, one panel per task |
| `figure7.py` | Full-model success with vs without a dynamic deadline |
| `figure8.py` | Semicircle-Narrow SWI / DTRL-On E1 / DTRL-On bars |
| `figure9.py` | Action inconsistency grouped by task |
| `figure10.py` | Pendulum warm-up ablation: E1/E2/E3 TensorBoard curves |
| `table2.py` | Render Table 2 in paper/booktabs layout (DTRL rows bold) |
| `table3.py` | Render Table 3 loss-weight allocation |
| `plot_all.py` | One-shot redraw of Tables 2–3 and Figures 6–10 from shipped data |
