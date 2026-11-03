# Random-instance 1,200-update run: raw fixed-panel curves

These are the **raw evaluation points** copied from
`formal_random_instances_s42`; no smoothing, no new rollouts, no new training.

- `diagnostic_curve.csv` — fixed diagnostic panel (`train10`, 10 instances)
- `validation_curve.csv` — fixed validation panel (`valid30`, 30 instances)
- `train_log.csv` — per-update training diagnostics (not a greedy panel curve)
- `diagnostic_greedy_curve.png` — raw diagnostic greedy mean, NN reference,
  validation-selected u1060 and final u1200 marked
- `validation_greedy_curve.png` — raw validation greedy mean, NN reference,
  validation best u1060 and final u1200 marked

Time is the model-estimated total time. The panels were already used during
development and review; they are not sealed test data.

## Fixed-panel values

| panel | initial | NN | validation best u1060 | final u1200 |
|---|---:|---:|---:|---:|
| diagnostic (10) | 3838.6 s | 3514.9 s | 3406.9 s | 3427.5 s |
| validation (30) | 3698.1 s | 3412.1 s | 3280.0 s | 3314.3 s |

The diagnostic panel's own minimum is at u1180 (3374.9 s); validation selection
uses `valid30` only.
