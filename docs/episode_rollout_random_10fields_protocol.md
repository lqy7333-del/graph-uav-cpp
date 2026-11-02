# Random-instance fixed-10-field complete-episode PPO protocol

**Status: implemented and tested; formal 600-update training NOT started.**

- Branch: `codex/episode-rollout-10fields`
- Review baseline: `e0a41fc`
- Previous code baseline: `911da56`
- **Training source: a new generated instance at every training episode.**
- **Fixed panels: `train10` is the diagnostic panel; `valid30` is the validation panel. Neither is used for training.**
- Initialization: random actor/critic, new optimizers, update 0. No B42/240/u780 load.
- Original u780 is missing and is neither sought nor recreated.

## 1. Corrected experiment target

This is **same-scale generalization**:

- every instance has exactly **10 fields**;
- each training episode starts from a newly generated field layout, shape,
  depot and configured geometry;
- the 5–20 field range is not sampled during training;
- the previous ten fixed training instances are no longer a training set; they
  are only the diagnostic evaluation panel.

It is **not** a fixed-ten-instance overfit experiment and **not** a completed
5–20-field unified generalization experiment. `valid30` measures same-scale
(10-field) new-instance behavior only.

## 2. Preserved training implementation

The already-reviewed complete-episode path is retained:

- 4 complete episodes/rank, 16 per update on 4 GPUs;
- safety cap 500 environment steps/episode;
- normal terminal bootstrap 0; truncated bootstrap retained and marked;
- failures/truncations counted, not replaced;
- per-episode GAE with no cross-episode leakage;
- padding masks excluded from losses/statistics;
- per-synchronized-mini-batch global valid-count normalization;
- global-zero mini-batch skips both optimizers on every rank;
- global advantage standardization;
- all-rank value/tail/log-prob/finite/replay/closure gates;
- checkpoint actor/critic/optimizers/RNG/rank runtime;
- recorded greedy actions and independent replay.

Network, action space, workload, auto-resupply, return logic, reward kernel and
maneuver proxy are unchanged.

## 3. Resolved training configuration

| item | value |
|---|---|
| instance source | **random_instances** |
| fields per instance | **10 exactly** |
| training seed base | **900,000,000** |
| rank blocks | 1,000,000 seeds; rank r starts at base + r×1,000,000 |
| seed stride | 64 per episode; retries use +attempt (max 20) |
| area range | **5,000–20,000 m²** (0.5–2.0 ha), inherited |
| collection mode | `complete_episodes` |
| episodes/rank | 4 |
| world size | 4 |
| updates | **600** = 9,600 training episodes |
| safety cap | 500 steps/episode |
| actor LR | **1e-4** |
| critic LR | **0.003** |
| LR schedule | **none** |
| gamma / GAE lambda | 1.0 / 0.95 |
| clip / epochs / batch | 0.2 / 4 / 64 |
| entropy schedule | 0.01, hold 0.1, end 0.0 |
| candidate_attention | false |
| randomize_uav | false |
| reward | time; existing maneuver proxy |
| optimizer weighting | transition-weighted PPO, not per-trajectory |
| eval | update 0 and every 10 updates, two fixed 10-field panels |
| training seed | 42 |

Resolved config and actual checkpoint metadata:
[`episode_rollout_resolved_config_random_instances.json`](episode_rollout_resolved_config_random_instances.json).

## 4. Instance generation and randomness

- A new `InstanceGenerator` call is made at each episode start with
  `num_fields=10`, `randomize_each_reset=False`, `randomize_uav=False`.
- Generator seed is `base + rank*1,000,000 + counter*64 + attempt`.
  It does not depend on episode length, actions, policy RNG or evaluation.
- Counter advances once per episode start, including failure/truncation.
- Rank ranges are disjoint by construction. The fixed panels use seeds
  81,000–81,009 and 82,000–82,029 and are outside the training range.
- Generation retries are bounded at 20 and recorded per episode; no map is
  selected by policy performance.
- After the fourth terminal episode, **no reset is performed**. The next update
  starts the next episode and generates its instance only then.
- Training does not pre-generate 9,600 maps; generation is on demand.

Every rank writes `train_episodes_rank_<rank>.csv` with:
update, rank, episode sequence, instance seed, fingerprint, field count, area
summary, generation attempts/retries, stream counter, completion/truncation/
failure, steps, total time/distance and coverage.

## 5. Checkpoint/resume

Checkpoint runtime includes:

- actor/critic/optimizer states and update count;
- policy RNG and rank runtime;
- instance stream base/rank/counter/retries;
- instance mode, field count, generator config hash;
- panel manifest hashes, world size, episodes/rank, budget, seed, max steps.

Resume is supported only at update boundaries. It validates collection mode,
instance mode, 10-field count, generator config hash, seed base, panels, world
size, episodes/rank, budget, seed and max steps. Continuous and resumed runs
must have the same later instance seeds/fingerprints. Mid-episode resume is not
supported.

## 6. Evaluation panels and curves

- `train10` → **fixed diagnostic panel** (10 fields, seeds 81000–81009).
- `valid30` → **fixed validation panel** (10 fields, seeds 82000–82029).
- Both are fixed, already used during development/review, and are **not
  sealed test** data.
- At update 0 and every 10 updates, greedy evaluation covers both panels.
- Each instance row includes cost, completion, actions and per-step summaries;
  independent replay must match completion state, termination reason and cost.
- Real failures are retained and completion rate is explicit. Means are shown
  only when all required rows completed.
- Validation `validation/best.json` selects checkpoints by the pre-registered
  completion-first, all-complete time rule; final and intermediate checkpoints
  remain.
- Diagnostic curve and validation curve are separate:
  `diagnostic/diagnostic_curve.csv`, `validation/validation_curve.csv`.
- Training random-instance sample statistics are separate:
  `train_log.csv` and `train_episodes_rank_*.csv`; they are never mixed with
  fixed-panel greedy curves.
- The final independent test set is not run and is not used for configuration
  selection.

## 7. Formal command (NOT executed)

New output directory, distinct from old fixed-bank runs:

```bash
cd /home/ps/Desktop/Graph_planning/graph-uav-episode-10fields
PYTHONPATH=. /home/ps/Desktop/Graph_planning/graph-uav-cpp/.venv/bin/python \
  -m coverage_env.rl.trainer.episode_random_worker \
  --diagnostic-bank /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/train10 \
  --validation-bank /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/valid30 \
  --output /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_instances_s42 \
  --num-updates 600 --episodes-per-rank 4 --seed 42 \
  --instance-seed-base 900000000 \
  --actor-lr 1e-4 --critic-lr 3e-3 \
  --device cuda --checkpoint-every 10
```

No `--initial-checkpoint` is supplied.

Boundary resume command (only after a checkpoint exists):

```bash
PYTHONPATH=. /home/ps/Desktop/Graph_planning/graph-uav-cpp/.venv/bin/python \
  -m coverage_env.rl.trainer.episode_random_worker \
  --diagnostic-bank .../train10 --validation-bank .../valid30 \
  --output .../formal_random_instances_s42_resume \
  --resume .../formal_random_instances_s42/ckpt_update_10.pt \
  --device cuda --checkpoint-every 10
```

## 8. Cost estimate

Random-instance 4-GPU smoke, 2-instance diagnostic + 2-instance validation
panels, 4 episodes/rank:

- 3 updates: **148 s total wall**;
- training accounting: **83.9 s** (≈28 s/update);
- 48 episodes / 954 transitions / 24 synchronized optimizer steps;
- 12 distinct seeds and 12 distinct fingerprints per rank; all exactly 10 fields.

Training-only extrapolation for 600: **≈4.7 h**. Evaluation scales to 40 panel
instances and 61 points; using the smoke eval share gives **≈2.5–3 h**. Budget
**≈7–8 h** before the formal run; this is not a measured 600-update time.

## 9. Tests

New random-instance tests: 5/5 pass. Full suite: **309 tests, 307 pass,
2 pre-existing failures, 3 skipped**. The two failures reproduce at baseline
`911da56`.

Evidence:

- [`episode_rollout_random_smoke.json`](episode_rollout_random_smoke.json)
- [`episode_rollout_random_resume.json`](episode_rollout_random_resume.json)
- [`episode_rollout_review_results.json`](episode_rollout_review_results.json)

**No formal 600-update run was started.**
