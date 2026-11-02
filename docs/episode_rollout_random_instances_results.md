# Random-instance fixed-10-field PPO: 1,200-update result

**Run status: completed. Final independent/sealed test not run.**

- Experiment: fresh 10-field instance at every training episode
- Start: random actor/critic, no B42/240 load
- Config: actor LR 1e-4, critic LR 3e-3, LR schedule none, gamma 1.0,
  GAE lambda 0.95, reward=time, area 5,000–20,000 m²
- Budget: 1,200 updates / 19,200 episodes / 356,680 transitions
- Checkpoints: every 10 updates; full greedy panels every 20; update 0 evaluated
- Start: 2026-11-02 21:29:31 +08:00
- End: 2026-11-03 06:53:11 +08:00
- Wall time: **9 h 23 m 40 s**
- Training accounting (`wall_seconds_max`): **7 h 39 m**
- Failures: **0**; truncations: **0**
- All instances had exactly **10 fields**
- Optimizer steps: **9,600** (8 per update)

## Fixed-panel results

These are fixed, already-used diagnostic/validation panels; they are **not
sealed test** and are not used for training.

| panel | initial (random) | best checkpoint | final update 1200 | final vs initial | final vs NN |
|---|---:|---:|---:|---:|---:|
| diagnostic `train10` (10) | 3838.6 s | **3374.9 s** (u1180) | 3427.5 s | **+411.0 s** | +87.4 s |
| validation `valid30` (30) | 3698.1 s | **3280.0 s** (u1060) | 3314.3 s | **+383.8 s** | +97.8 s |

- Validation-selected best: **update 1060**, 3280.0 s.
- Final update 1200 is 34.3 s slower than best validation, but remains 383.8 s
  faster than the random initial model.
- Diagnostic best is update 1180; diagnostic is observation-only.
- Final vs initial: **30/30 validation instances improved**, 10/10 diagnostic.
- Final vs NN: **27/30 validation** and **7/10 diagnostic** instances improved.
- All validation/diagnostic rows completed; no success-only mean was used.

### Time components

Validation panel:

| component | initial | final | change |
|---|---:|---:|---:|
| work base | 1467.7 s | 1466.1 s | −1.6 s |
| nonwork | 1837.0 s | 1494.8 s | **−342.2 s** |
| maneuver | 393.4 s | 353.4 s | **−40.0 s** |
| total | 3698.1 s | 3314.3 s | **−383.8 s** |

Most of the improvement comes from nonwork transfer/return time, with a smaller
maneuver reduction; necessary work time is essentially unchanged.

## Training outcome

- 4 rank streams stayed disjoint; every rank generated 4,800 distinct training
  instances.
- No gate failure, non-finite update, failed episode, or safety truncation.
- The validation curve improved to the best at 1060 and then did not exceed the
  final 1200 value on the validation-selection rule; final is retained.
- This is a single training seed. It is **same-scale (10-field) new-instance
  evidence**, not a 5–20-field unified generalization result and not a sealed
  test.

## Artifacts

Root:

`/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_instances_s42/`

- `ckpt_update_1060.pt` — validation-selected best
- `ckpt_update_1200.pt` — final
- `diagnostic/evaluation/*.json`, `diagnostic/diagnostic_curve.csv`
- `validation/evaluation/*.json`, `validation/validation_curve.csv`
- `train_log.csv`
- `train_episodes_rank_0..3.csv`
- `run_summary.json`
- `generation_retries_rank_*.jsonl` if any occurred (none expected)

JSON summary: [`episode_rollout_random_instances_results.json`](episode_rollout_random_instances_results.json).
