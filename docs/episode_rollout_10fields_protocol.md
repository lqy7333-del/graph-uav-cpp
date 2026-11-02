# Fixed 10-field complete-episode PPO protocol

**Status: review fixes implemented and tested; random-init formal training NOT started.**

- Branch: `codex/episode-rollout-10fields`
- Review baseline: `2d44e57`; fixes after that commit
- Baseline code: `911da56` (`codex/fixed-instance-overfit`)
- **Initialization: random actor and critic; no B42/240 weights**
- Original u780: not sought, not retrained, not substituted.

## Why the mode exists

`DistributedTrainer.train` resets the environment at every update and re-resets
after the last step of a batch. That is correct for the legacy fixed-step
rollout but cannot continue a live episode across a batch boundary. The new
`EpisodeRolloutTrainer` collects exactly `ppo.episodes_per_rank` terminal
episodes per rank before one synchronized PPO update. A fourth episode that
ends at the budget boundary is **not** followed by a reset; the next update
starts with a reset only because a new episode is required. The legacy
`fixed_steps` path is unchanged and remains the default.

## Resolved protocol

| item | value |
|---|---|
| collection mode | `complete_episodes` |
| episodes per rank | 4 |
| GPUs / world size | 4 |
| episodes per synchronized update | 16 |
| safety cap | 500 environment steps/episode |
| start | **random actor/critic**, new optimizers, update 0, new schedules |
| actor LR | **1e-4** |
| critic LR | **0.003** |
| gamma / GAE lambda | 1.0 / 0.95 |
| clip / epochs / batch | inherited: 0.2 / 4 / 64 |
| entropy schedule | inherited: 0.01, hold 0.1, end 0.0 |
| LR schedule | `none` |
| candidate_attention | false |
| reward | time, existing maneuver proxy |
| randomize_uav | false |
| optimizer protocol | transition-weighted PPO (not per-trajectory weighting) |
| update budget | **600** = 9,600 sampled episodes (not a convergence claim) |
| checkpoint/eval | update 0 and every 10 updates |
| training seed | 42 |

The complete resolved config and a field-by-field diff are in
[`episode_rollout_resolved_config_random.json`](episode_rollout_resolved_config_random.json).
Relative to the bank configuration, the only real change is actor LR
`3e-4 → 1e-4`; critic LR remains `0.003`, and no reward, feature, network,
executor, workload, resupply or return logic changes. LR schedule is explicitly
`none`.

## Fixed banks

All banks are 10 fields, generated from the B42/240 config with the original
0.5–2.0 ha area range. Full geometry and initial environment state are stored in
`instances.pt`; only identity/checksums are in Git.

| bank | purpose | seeds | instances | manifest SHA256 prefix | instances SHA256 prefix |
|---|---|---|---:|---|---|
| `train10` | training | 81000–81009 | 10 | `a46c00d5ee50ca8e` | `4bd18aff19c649d9` |
| `valid30` | validation only | 82000–82029 | 30 | `72d1ae92c73a8fc3` | `3c7b5ed5663f8798` |
| `smoke_bank` | temporary smoke only | 83000–83001 | 2 | `8dfceb8778d48c60` | `8b085aa3947623b4` |

Persistent root:

`/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/`

These seed blocks are disjoint from the historical 40000–40095, 50000–51599,
60000–61599 and 70000–70009 blocks. Manifest, snapshots, SHA256 and instance
fingerprints are in [`episode_rollout_banks.json`](episode_rollout_banks.json).

## Semantics

- Sampling network is in eval/deterministic mode; policy parameters fixed during
  each collection.
- Per-episode GAE; normal termination sets `next_value=0`; no GAE leaks across
  episodes.
- Safety truncation preserves the legal bootstrap and marks `truncated=True`;
  it is not counted as a successful complete task and aborts that update with a
  diagnostic (`safety truncation ...`). No retry-to-fill-success loop.
- Real failures are counted; they are not dropped and replaced.
- Unequal local transition counts are padded only for DDP synchronization.
  Padding rows have mask 0 and are excluded from policy loss, value loss,
  entropy, advantages and statistics.
- Every rank performs the same number of backward/optimizer steps; loss is
  scaled by the all-reduced global valid-transition count and DDP world size.
- Advantages are standardized with all-reduced global valid sum/sumsq/count,
  not per-rank means.
- No duplicate real samples are counted; local zero-valid rank is not possible
  under `episodes_per_rank >= 1` (the collector refuses zero episodes).
- Training bank order is a resumable shuffled-without-replacement cycle seeded
  per rank (`FixedInstancePool`); bank id/index/seed never enter observations.
- `FixedInstancePool.reset()` restores pristine geometry, resources and heading;
  fingerprint checked on every reset.

## Evaluation

- Greedy fixed-bank evaluation at update 0 and every 10 updates.
- Training bank curve and validation bank curve are written separately:
  `learning_curve.csv` and `validation/validation_curve.csv`.
- Evaluation saves full action lists and per-step state summaries; each row is
  replayed from a pristine environment and must match completion/time before it
  is written (`replay_ok=true`).
- Evaluation captures/restores Python/NumPy/Torch RNG and does not consume the
  collection pool's order or optimizer state.
- Failed instances are not dropped; completion rate is reported. Mean cost is
  populated only when all requested rows completed (same `aggregate_rows`
  contract as the overfit evaluation).

## Resume scope

Supported: **exact continuation at a saved update boundary**. A CPU test
(2 episodes/rank, 2 updates) compared a continuous update-2 checkpoint with a
checkpoint resumed from update 1: actor and critic tensors were bit-identical.
The checkpoint stores actor/critic/optimizers, RNG, rank runtime/environment and
update count. Resume validates world size, episodes per rank, update budget,
seed, collection mode, training bank and validation bank before continuing.
Same-directory resume is tested. Mid-episode resume is **not** supported and not
claimed.

## Formal command (NOT executed)

```bash
cd /home/ps/Desktop/Graph_planning/graph-uav-episode-10fields
PYTHONPATH=. /home/ps/Desktop/Graph_planning/graph-uav-cpp/.venv/bin/python \
  -m coverage_env.rl.trainer.episode_worker \
  --bank /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/train10 \
  --validation-bank /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/valid30 \
  --output /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_s42 \
  --num-updates 600 --episodes-per-rank 4 --seed 42 \
  --actor-lr 1e-4 --critic-lr 3e-3 \
  --device cuda --checkpoint-every 10
```

Boundary resume command (only after a checkpoint exists):

```bash
PYTHONPATH=. /home/ps/Desktop/Graph_planning/graph-uav-cpp/.venv/bin/python \
  -m coverage_env.rl.trainer.episode_worker \
  --bank /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/train10 \
  --validation-bank /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/valid30 \
  --output /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_s42_resume \
  --resume /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_s42/ckpt_update_10.pt \
  --device cuda --checkpoint-every 10
```

## Cost estimate (conditional, from 3-update smoke)

Random-init smoke was 2 training instances + 2 validation instances, 4 GPUs,
4 episodes/rank: **3 updates took 122 s total wall**, of which training
accounting was **54.9 s** (≈18.3 s/update). Training-only extrapolation:
600 updates ≈ **3.0 h**.

Evaluation cost scales with 40 instances/checkpoint (10 train + 30 validation)
and 61 evaluation points (0,10,…,600); smoke had 4 instances/checkpoint and
4 points. Using the smoke eval share gives a conservative **~2.5–3 h** for
evaluation. Therefore the formal run should be budgeted at **roughly 5–6.5 h**,
pending the reviewer's acceptance. This is an estimate, not a measured
600-update time.

## Tests

See [`episode_rollout_implementation_review.md`](episode_rollout_implementation_review.md).
