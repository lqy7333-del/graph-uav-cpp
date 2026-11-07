# CHANet component ablations C/D — final results

**Single paired training seed42. No sealed test. No A/B rerun. No added seeds.**

- A: full CHANet, `formal_random_instances_s42`
- B: `node_mlp`, existing result
- C: `no_cross_field_boundary` — `BoundaryInterEncoder` replaced
- D: `no_intra_boundary` — `BoundaryIntraEncoder` replaced
- C/D trained from A `ckpt_update_0.pt`; all retained modules copied, only the
  replaced layer randomly initialized; new optimizers.
- A update0 SHA256: `114d11190d1282edc9377db2f846244a1420792a9cf11542ee3553189f7443db`
- Baseline/code commit for C/D: `7392947`

## Validation-selected and final results

Model selection is on the **fixed validation panel only**. Diagnostic best is
never used to select a model.

| model | validation best | time (s) | final u1200 (s) |
|---|---|---:|---:|
| A full CHANet | u1060 | **3280.0** | **3314.3** |
| B node_mlp | u760 | 3383.6 | 3397.3 |
| C no cross-field boundary | u1020 | 3335.8 | 3348.4 |
| D no intra-boundary | u900 | 3310.7 | 3352.2 |
| NN reference | — | 3412.1 | 3412.1 |

All fixed-panel rows were complete: C/D **30/30** validation, 10/10 diagnostic.

## C/D versus A

Difference is **ablation − A**; **positive means A (full CHANet) is better**.

| comparison | difference | fixed-30 conditional 95% CI | A wins / ablation wins |
|---|---:|---:|---:|
| C selected: C u1020 − A u1060 | **+55.79 s** | **[+26.76, +86.79]** | **22 / 8** |
| C final: C u1200 − A u1200 | **+34.11 s** | [−0.11, +68.68] | 19 / 11 |
| D selected: D u900 − A u1060 | **+30.69 s** | **[+5.98, +55.21]** | **22 / 8** |
| D final: D u1200 − A u1200 | **+37.85 s** | **[+18.51, +58.60]** | **22 / 8** |

- C is slower than A at both validation-selected and final points; the selected
  gap is clear, the final gap is borderline on this fixed panel.
- D is slower than A at both points; both intervals exclude zero.
- C and D are separate component ablations. Do **not** add their differences as
  an additive decomposition of CHANet; components can interact.

## Time components

| model / point | work | nonwork | maneuver | service | total |
|---|---:|---:|---:|---:|---:|
| A best u1060 | 1465.4 | 1467.5 | 347.0 | 0 | 3280.0 |
| C best u1020 | 1466.1 | 1508.3 | 361.3 | 0 | 3335.8 |
| D best u900 | 1466.5 | 1488.6 | 355.5 | 0 | 3310.7 |
| A final u1200 | 1466.1 | 1494.8 | 353.4 | 0 | 3314.3 |
| C final u1200 | 1468.3 | 1516.4 | 363.7 | 0 | 3348.4 |
| D final u1200 | 1465.7 | 1523.5 | 363.0 | 0 | 3352.2 |

C/D's extra cost is again mostly nonwork, with smaller maneuver increases;
required work time is nearly equal.

## Diagnostic panel (observation only)

| model | diagnostic minimum | final u1200 |
|---|---:|---:|
| A | u1180, 3374.9 | 3427.5 |
| B | u760, 3468.0 | 3501.6 |
| C | u1200, 3437.5 | 3437.5 |
| D | u820, 3415.1 | 3458.5 |

These are not used for checkpoint selection.

## Raw greedy curves

All four raw curves and NN are shown together; no smoothing.

![validation A/B/C/D](/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/chanet_component_ablation/validation_ABCD_curve.png)

Files:

- `A_validation_curve.csv`, `B_validation_curve.csv`,
  `C_validation_curve.csv`, `D_validation_curve.csv`
- same four diagnostic CSVs
- `A_train_log.csv` … `D_train_log.csv`
- `validation_ABCD_curve.png`, `diagnostic_ABCD_curve.png`

## Model size and inference

| model | actor params | encoder params | replaced module params | median inference |
|---|---:|---:|---:|---:|
| A | 1,252,227 | 983,296 | 232,320 | 10.25 ms |
| B | 1,251,561 | 982,630 | — | 8.65 ms |
| C | 1,252,105 | 983,174 | 232,198 | 9.97 ms |
| D | 1,252,105 | 983,174 | 232,198 | 10.05 ms |

C/D differ from A by **−122 params** in encoder/actor (−0.0124%), with hidden
width 902 for the local replacement. Inference is one fixed 10-field validation
instance, 20 warmups + 200 timed GPU forwards, median wall time. Latency is not
a substitute for training compute.

## Training volume and wall time

| run | start | end | wall | training accounting | episodes | transitions | failures/trunc |
|---|---|---|---:|---:|---:|---:|---:|
| C | 2026-11-06 22:18:38 | 2026-11-07 07:20:11 | **8 h 01 m 33 s** | 7 h 18 m 43 s | 19,200 | 347,760 | 0 / 0 |
| D | 2026-11-07 07:20:11 | 2026-11-07 16:29:25 | **9 h 09 m 14 s** | 7 h 25 m 16 s | 19,200 | 352,477 | 0 / 0 |

Both use 4 GPUs, 4 complete episodes/rank, 16/update, 1,200 updates, 9,600
synchronized optimizer steps, and a fresh 10-field instance for every episode.
No gate failure, NaN/Inf, unused-parameter DDP error, or truncation occurred.

Smoke evidence:

- C/D 3-update smoke: 12 episodes/rank, first 12 seed/fingerprints identical to A;
- resume update3→4 in independent output: variant/paired record/stream preserved,
  rows 13–16 match A, parameters update, gates pass;
- 50 targeted tests passed before formal launch.

## Best/final checkpoint locations

- C best: `episode10/ablation_no_inter_s42/ckpt_update_1020.pt`
- C final: `episode10/ablation_no_inter_s42/ckpt_update_1200.pt`
- D best: `episode10/ablation_no_intra_s42/ckpt_update_900.pt`
- D final: `episode10/ablation_no_intra_s42/ckpt_update_1200.pt`
- A best/final: `episode10/formal_random_instances_s42/ckpt_update_{1060,1200}.pt`
- B best/final: `episode10/ablation_node_mlp_s42/ckpt_update_{760,1200}.pt`

Full SHA256s, per-instance rows and components are in `comparison.json`.

## Boundary

This is one paired training seed. The intervals are conditional on the fixed
30-instance validation panel. C/D differences are not additive; they are two
separate structural ablations. No sealed test, new training seed or extra budget
was run.
