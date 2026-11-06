# CHANet whole-encoder ablation: A vs B, seed42

**Single paired training seed. No sealed test. No A rerun.**

- A: full relation-aware graph encoder, `formal_random_instances_s42`
- B: `node_mlp`, no message passing/edge attributes; shared initial weights
  and critic from A update0; B-only MLPs randomly initialized; new optimizers
- Both: 1,200 updates, 4 complete episodes/rank, fresh random 10-field instance
  each episode, actor LR 1e-4, critic LR 3e-3, time reward, gamma1
- A update0 SHA256: `114d11190d1282edc9377db2f846244a1420792a9cf11542ee3553189f7443db`
- A encoder: 983,296 params; B encoder: 982,630 (−666, −0.068%)
- A actor total: 1,252,227; B: 1,251,561
- A wall time: 9 h 23 m 40 s; training accounting 7 h 39 m
- B wall time: 8 h 40 m 57 s; training accounting 6 h 56 m
- A transitions: 356,680; B: 348,503
- Both: 19,200 episodes, 9,600 optimizer steps, 0 failures, 0 truncations

## Fixed validation panel (valid30, n=30)

| quantity | A | B | B − A |
|---|---:|---:|---:|
| validation-selected checkpoint | **u1060** | **u760** | — |
| selected total time | **3280.0 s** | 3383.6 s | **+103.6 s** |
| final u1200 | **3314.3 s** | 3397.3 s | **+83.0 s** |
| NN reference | 3412.1 s | 3412.1 s | — |

`B − A` positive means **A is better**.

Selected-checkpoint paired comparison:

- B − A: **+103.61 s**, fixed-30 conditional 95% CI **[+63.25, +142.93]**
- A faster on **24/30** instances, B faster on **6/30**, 0 ties
- Completion: A **30/30**, B **30/30**

u1200 comparison:

- B − A: **+82.96 s**, fixed-30 conditional 95% CI **[+47.12, +117.53]**
- A faster on **24/30**, B faster on **6/30**, 0 ties
- Completion: A **30/30**, B **30/30**

### Validation time components

| component | A selected | B selected | B − A |
|---|---:|---:|---:|
| work base | 1465.4 s | 1465.8 s | +0.4 s |
| nonwork | 1467.5 s | 1554.2 s | **+86.7 s** |
| maneuver | 347.0 s | 363.6 s | +16.6 s |
| total | 3280.0 s | 3383.6 s | **+103.6 s** |

At final u1200: work +3.4 s, nonwork +72.3 s, maneuver +7.3 s, total +83.0 s.
The B disadvantage is mostly nonwork time, not the necessary work base.

## Fixed diagnostic panel (train10, n=10; observation only)

| quantity | A | B | B − A |
|---|---:|---:|---:|
| panel minimum | u1180, 3374.9 s | u760, 3468.0 s | +93.1 s |
| final u1200 | 3427.5 s | 3501.6 s | +74.1 s |

Diagnostic is **never used for checkpoint selection**. Selected-checkpoint
B−A is +93.14 s [63.40, 121.41], A 9/10, B 1/10. u1200 B−A is +74.08 s
[16.07, 141.66], A 8/10, B 2/10.

## Raw curves

- `A_validation_curve.csv`, `B_validation_curve.csv`
- `A_diagnostic_curve.csv`, `B_diagnostic_curve.csv`
- `A_train_log.csv`, `B_train_log.csv`
- `validation_AB_curve.png`, `diagnostic_AB_curve.png`

![validation A/B](/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/chanet_ablation/validation_AB_curve.png)

All plotted points are raw evaluation points; no smoothing replaces the curves.
Times are model-estimated greedy total times.

## Conclusion boundary

In this single paired seed, removing graph message passing/edge encoding (B)
produced a validation-selected model about **103.6 s slower** than A, and B
u1200 about **83.0 s slower**. The B disadvantage appears mainly in nonwork
time. This is one paired training seed with fixed panels; it is not a
training-seed significance claim, not a sealed-test result, and not a 5–20-field
generalization claim.

Model files remain in the persistent experiment directories:

- A: `episode10/formal_random_instances_s42/ckpt_update_1060.pt`, `ckpt_update_1200.pt`
- B: `episode10/ablation_node_mlp_s42/ckpt_update_760.pt`, `ckpt_update_1200.pt`

Machine-readable comparison: [`comparison.json`](comparison.json).
