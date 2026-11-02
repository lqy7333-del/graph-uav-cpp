# Complete-episode PPO implementation review

## Files

- `coverage_env/rl/trainer/episode_rollout.py` — complete-episode collector,
  per-episode GAE, global advantage normalization, masked PPO update.
- `coverage_env/rl/trainer/episode_random.py` — fresh random 10-field instance
  source, seed stream, diagnostic/validation panel evaluation.
- `coverage_env/rl/trainer/episode_random_worker.py` — random-instance DDP CLI;
  random initialization and boundary resume.
- `coverage_env/rl/trainer/episode_worker.py` — legacy fixed-bank CLI retained
  for historical compatibility.
- `coverage_env/rl/ppo/ppo.py` — `update_masked` for unequal local lengths.
- `coverage_env/rl/evaluation/maneuver_eval.py` — recorded actions, per-step
  summaries, independent replay helper.
- `coverage_env/rl/trainer/overfit.py` — evaluation requests recordings and
  compares replay state/cost.
- `coverage_env/config/default.yaml` — explicit collection-mode defaults;
  legacy `fixed_steps` remains default.
- `coverage_env/tests/test_episode_rollout.py` — 12 bounded tests.
- `scripts/maneuver_time/ddp_failure_ddp.py` — real-DDP Gloo/NCCL fault injection.
- `scripts/maneuver_time/ddp_gradient_smoke.py` — unequal-rank gradient reference.

## Preserved audited behavior

No reward, feature, network, executor, workload, resupply, return, GAE or gate
code was replaced. The existing GAE/PPO, graph encoder, masked decoder, time
ledger, fixed bank, checkpoint payload, DDP gate, rank-runtime and greedy
evaluator are reused.

Verified in the baseline and preserved:

- bootstrap uses real `next_state` including heading;
- true terminal `next_value=0`;
- done/truncated boundaries do not propagate GAE across episodes;
- actor eval mode for sampling and replay;
- all-rank value/tail/log-prob/finite gates and common abort;
- reward/time/distance/resource closure;
- checkpoint actor/critic/optimizers/RNG/rank runtime;
- fixed-bank manifest/fingerprint checks;
- greedy fixed-bank evaluation;
- recorded action replay for evaluation.

## Unequal DDP design

1. Each rank collects exactly 4 terminal episodes; local lengths differ.
2. Local GAE is computed once; episode boundaries prevent cross-episode leakage.
3. Global valid count/sum/sumsq are all-reduced; local advantages are
   standardized with the global mean/variance.
4. `global_steps = all_reduce_max(local_count)`; every rank pads to the same
   length and runs the same number of minibatches/backwards.
5. Padding rows have mask 0 and are excluded from policy loss, value loss,
   entropy, advantages and statistics.
6. **Each synchronized mini-batch is normalized by that mini-batch's global
   valid count**, not by the full rollout count. This preserves ordinary PPO's
   per-minibatch scale.
7. A mini-batch whose global valid count is 0 is skipped by **all ranks**; both
   optimizers are not stepped, so Adam momentum cannot move padding-only batches.
8. Logged `updates` counts actual synchronized optimizer steps.
9. Production `episodes_per_rank >= 1` guarantees each rank has at least one
   transition to use as a structurally valid padding template. The CLI rejects
   zero episodes; duplicate samples are never counted as valid.

## Interpretation boundary

The formal protocol trains on **freshly generated instances with exactly 10
fields** at every episode. `train10` is now a fixed diagnostic panel and
`valid30` is the fixed validation panel; neither is a training source. This is
same-scale generalization, not fixed-ten-instance overfitting and not a
completed 5–20-field unified generalization study.

## Evaluation semantics

Rows are compared with independent replay by `completed`, `termination_reason`
and `total_time`. A real failure is retained as a failed row; it is not coerced
to success, deleted, or replaced. Completion rate is reported separately, and
mean cost is filled only when all rows completed.

## Resume scope and checks

Supported: **exact continuation at a saved update boundary**, including
same-directory resume. Checkpoint payload stores actor/critic/optimizers, RNG,
rank runtime/environment, bank manifest and update count.

Resume rejects mismatches in:

- world size;
- episodes per rank;
- update budget;
- seed;
- `max_episode_steps`;
- collection mode;
- training bank manifest;
- validation bank manifest, omission, or addition;
- initialization source.

When resuming to a different output directory, references, existing evaluation
directories, learning/validation curves and best files are copied; same-directory
resume does not copy a file onto itself. Mid-episode resume is not supported and
not claimed.

## Real-DDP fault injection

`scripts/maneuver_time/ddp_failure_ddp.py` wraps **actor and critic in
DistributedDataParallel** and runs the production `EpisodeRolloutTrainer`.

Injections:

- rank 1 raises on its second sampling step, **after its first DDP policy
  forward**, while other ranks continue sampling;
- rank 1 raises gate construction after collection.

Results:

- 2-rank Gloo collect: both ranks raise, `update_count=0`, no `train_log.csv`,
  rank0 sampled 18 steps, rank1 2; diagnostic written.
- 2-rank Gloo gate: both ranks raise, no update, diagnostic written.
- **4-GPU NCCL collect**: all four ranks raise, all `update_count=0`, no log,
  rank0/2/3 sampled 18–19 steps while rank1 failed at step 2; diagnostic written.
- **4-GPU NCCL gate**: all four ranks raise, no update, diagnostic written.

Raw results: [`episode_rollout_review_results.json`](episode_rollout_review_results.json).

## Unequal-rank gradient reference

2-rank Gloo, actor/critic DDP-wrapped, no optimizer steps:

- rank0 3 valid transitions, rank1 6;
- batch size 4, global steps 6;
- rank0's second mini-batch is pure padding;
- global valid counts 7 and 2.

DDP accumulated gradients vs centralized reference:

| quantity | max difference |
|---|---:|
| actor | **5.96e-08** |
| critic | **2.98e-08** |

## Test evidence

### Unit/CPU

`test_episode_rollout.py`: **12/12 pass**:

- episode length exceeds ignored legacy `rollout_steps`;
- exactly N starts, no extra reset/zero-step instance at batch boundary;
- next update starts cleanly;
- isolated cross-episode GAE hand calculation;
- truncated bootstrap preserved while GAE breaks;
- masked padding excluded from actor statistics;
- mini-batch loss scale `-1` restored (not `-0.5`);
- global padding-only mini-batch does not move parameters;
- injected collect/gate failures write diagnostics;
- evaluation action replay;
- evaluation failure replay compared rather than coerced to success.

### Regression/full suite

Affected regression subset: 44 pass, 3 skipped.

Full suite:

`python -m unittest discover -s coverage_env/tests -p 'test_*.py'`
→ **304 tests, 302 pass, 2 failures, 3 skipped**.

The two failures reproduce at baseline `911da56`:

- `test_nearest_neighbor_episode...test_5_10_20_fields_x_10_episodes`;
- `test_uav...test_depot_and_manual_path_execution`.

### Random-instance 4-GPU smoke

3 updates with fresh generated 10-field instances; the fixed panels used for
evaluation were the 2-instance smoke bank:

- **148 s total wall**;
- training accounting **83.9 s** (≈28 s/update);
- 48 episodes / 954 transitions;
- 24 synchronized optimizer steps;
- 12 distinct seeds and 12 distinct fingerprints per rank; every instance has
  10 fields;
- actual actor/critic LR `1e-4 / 0.003`; `initialization=None`;
- reward mode `time`, gamma `1.0`, area range **5,000–20,000 m²**;
- no failure/truncation; diagnostic/validation greedy replay checks pass.

Random stream/resume evidence:
[`episode_rollout_random_smoke.json`](episode_rollout_random_smoke.json),
[`episode_rollout_random_resume.json`](episode_rollout_random_resume.json).

### Exact boundary resume

CPU continuous run vs resume from update-1 checkpoint: update-2 actor/critic
state dicts bit-identical. Wrong `max_episode_steps` is rejected; same-directory
resume passes.

### Random-instance tests

`test_random_instance_mode.py`: **5/5 pass**:

- rank seed blocks disjoint and reproducible;
- stream counter independent of action-length randomness;
- fresh instances change seed/fingerprint and all have 10 fields;
- fixed panel evaluation does not advance the stream or training RNG;
- rank-local episode records preserve 10-field identity.

Full suite is **309 tests, 307 pass, 2 failures, 3 skipped**; the two failures
reproduce at baseline `911da56`.

### Not run

- Formal 600-update random-instance training.
- Final/sealed test set.
- Mid-episode resume.
- Complete `RUN_DDP_TESTS=1` suite; targeted real-DDP Gloo and 4-GPU NCCL
  injection and gradient scripts were run directly.
