# Complete-episode PPO implementation review

## Files

- `coverage_env/rl/trainer/episode_rollout.py` — complete-episode collector,
  per-episode GAE, global normalization, masked update, train/validation eval.
- `coverage_env/rl/trainer/episode_worker.py` — fixed-bank DDP worker and CLI,
  new-optimizer fine-tune and boundary resume.
- `coverage_env/rl/ppo/ppo.py` — `update_masked` for unequal local lengths.
- `coverage_env/rl/evaluation/maneuver_eval.py` — optional recorded action
  sequence, per-step summaries and independent replay helper.
- `coverage_env/rl/trainer/overfit.py` — evaluation now requests recordings and
  refuses a row whose replay disagrees.
- `coverage_env/config/default.yaml` — explicit `collection_mode`,
  `episodes_per_rank`, `max_episode_steps` defaults; legacy remains default.
- `coverage_env/tests/test_episode_rollout.py` — new bounded tests.

## Preserved audited behavior

The implementation reuses the existing GAE/PPO, graph encoder, masked decoder,
reward/time ledger, fixed bank, checkpoint payload, DDP gate, rank-runtime and
greedy evaluator. No reward, feature, network, executor, workload, resupply or
return code was changed.

The pre-existing diffs at B42/240 that matter were checked in the baseline and
not overwritten:

- GAE bootstrap next state includes heading and real `next_state`;
- true terminal `next_value=0`;
- done/truncated boundaries are separate in `compute_advantage`;
- actor eval mode is used for both sampling and replay;
- all-rank value/tail/replay/finite gates;
- all-rank abort and failure snapshots;
- reward/time/distance/resource ledger;
- checkpoint actor/critic/optimizers/RNG/rank runtime;
- fixed-bank manifest/fingerprint checks;
- greedy fixed-bank evaluation;
- recorded action replay is now wired into evaluation.

## Unequal DDP design

1. Each rank collects exactly 4 terminal episodes, so local lengths differ.
2. Local GAE is computed once on the complete buffer; episode boundaries prevent
   cross-episode propagation.
3. Global valid count/sum/sumsq are all-reduced; local advantages are standardized
   with the global mean/variance.
4. `global_steps = all_reduce_max(local_count)` and each rank pads its local
   transition list to `global_steps`. Every rank runs the same number of
   minibatches/backwards.
5. Padding rows are structurally valid repeats of the last local transition but
   `mask=0`; they are excluded from all losses/statistics.
6. Loss terms are summed over valid local rows and multiplied by
   `world_size/global_valid_count`, compensating for DDP gradient averaging.
7. The critic loss uses the same valid mask and normalization.
8. A rank cannot have zero valid rows under `episodes_per_rank >= 1`; the CLI
   rejects zero. This is stated rather than silently injecting duplicate data.

## Known limitations

- **Only boundary resume is supported.** A checkpoint is saved after an update;
  the fourth episode's terminal state is saved in rank runtime. The next update
  starts with reset. Mid-episode/arbitrary-time resume is not claimed.
- The `update_masked` API requires at least one local transition, because it needs
  a structurally valid padding state. Production collection guarantees that.
- Formal 200-update training and the final test set were not run.
- Validation uses the same greedy decoding/evaluation as training-bank eval; it
  is not a learned early-stopping signal.
- Validation writes `validation/best.json` using the existing completion-first,
  all-instances-complete time rule. Final checkpoints and all intermediate
  checkpoints are preserved; selection must not be redefined after seeing results.

## Test evidence

### Unit/CPU

`test_episode_rollout.py`: 7 tests, all pass:

- complete episodes exceed the ignored legacy `rollout_steps`;
- exactly N starts and no extra reset/zero-step instance at the batch boundary;
- next update starts a new episode cleanly;
- isolated GAE hand calculation across two episodes;
- truncated bootstrap preserved while GAE breaks;
- masked padding excluded from actor statistics;
- independent recorded replay of evaluation actions;
- evaluation writes actions/state summaries/replay_ok.

### Existing regression subset

`test_fixed_instance_overfit`, `test_maneuver_eval`, `test_training_ddp_runtime`
+ new tests: 39 tests pass (3 skipped without `RUN_DDP_TESTS=1`).

### Full suite

`python -m unittest discover -s coverage_env/tests -p 'test_*.py'`:
**299 tests, 297 pass, 2 failures, 3 skipped**.

The two failures are pre-existing at baseline `911da56` and reproduce there:

- `test_nearest_neighbor_episode...test_5_10_20_fields_x_10_episodes`
  (`0 != 10` completed, configuration exception/payload edge);
- `test_uav...test_depot_and_manual_path_execution` (`16.0 != 40.0`).

They are not caused by this branch and were not silently fixed.

### 4-GPU smoke

Command used a dedicated 2-instance smoke bank and a separate output dir:

```bash
python -m coverage_env.rl.trainer.episode_worker \
 --bank .../smoke_bank --validation-bank .../smoke_bank \
 --initial-checkpoint .../initial_B42_240.pt \
 --output .../gpu_smoke_final \
 --num-updates 3 --episodes-per-rank 4 --seed 42 \
 --device cuda --checkpoint-every 1
```

Result: exit 0, wall **119 s**; 3 updates; rank0 12 episodes / 214 transitions;
global 48 episodes / 868 transitions; 24 synchronized optimizer steps;
no failed or truncated episodes; gates passed; train and validation eval wrote
recorded actions and replay_ok=true for each row.

### Exact boundary resume

CPU full run (2 updates, 2 episodes/rank) vs resume from update-1 checkpoint:
update-2 actor and critic state dicts are bit-identical. This validates
optimizer, RNG, bank order and schedule continuation at an update boundary.

### Not run

- Formal 200-update training.
- Final/sealed test set.
- Mid-episode resume.
- `RUN_DDP_TESTS=1` loopback Gloo suite (the real 4-GPU NCCL smoke was run).
