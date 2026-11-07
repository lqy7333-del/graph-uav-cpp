# Random 30-field full-episode CHANet, seed42

Status: implementation, geometry audit and real 4-GPU smoke complete.  
**Budget authorization accepted 2026-11-07:** the user explicitly approved the
measured 40–42-hour wall clock and removed the earlier 36-hour auto-start cap.
Formal training is authorized and running; this document records the frozen
configuration, authorization, launch SHA and measured first-20-update timing.

## Branch and baseline

- Branch: `codex/episode-rollout-30fields`
- Baseline: `27a9d4f` (`codex/episode-rollout-10fields` latest at branch creation)
- Existing A/B/C/D code, models, manifests and results are preserved.
- No extra ablation, training seed, or sealed evaluation is added.

## Geometry audit (100 candidate seeds)

Same workspace and field generator as the 10-field A run:

- workspace: lat 31.21–31.24, lon 121.455–121.485 (≈9.54 km²)
- field area: 5,000–20,000 m²
- tessellated layout, parcel gap 8 m; UAV speed 8 m/s; coverage width 10 m
- 30 fields per instance

Candidates **1,300,000–1,300,099**:

| item | result |
|---|---:|
| success | **97 / 100** |
| failures | 3 |
| success rate | **97%** |
| failed seeds | 1,300,019, 1,300,043, 1,300,098 |
| failure cause | 8 internal placement attempts, workspace too crowded |
| valid field count | 30/30 for every accepted instance |
| total field area | 305,981–383,447 m² |
| individual area | 5,000–19,900 m² |
| maximum pairwise overlap | **0.0 m²** |
| minimum pairwise gap | **8.0 m** |
| outside workspace | **0** |

The 97% success rate is above the 90% preflight threshold. Failures are recorded
and only expected `GeometryGenerationError`s are retried; no field size/area
range/workspace change and no policy-based seed screening was used.

Fixed seed assets (new namespaces, no training overlap):

- diagnostic panel: seeds **1,310,000–1,310,009**, 10 fixed 30-field instances;
  generation failures 0
- validation panel: seeds **1,320,000–1,320,029**, 30 fixed 30-field instances;
  generation failures 0
- training base: **1,330,000,000**, rank blocks 1M, stride64 per episode
- candidate audit block: 1,300,000–1,300,099

Full snapshots and manifests:

`/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode30/diagnostic10/`  
`/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode30/validation30/`

Panels are never sampled by the training stream.

## Frozen training configuration

Derived from the 10-field A checkpoint config, changing only field count and
path/stream labels:

- encoder: complete CHANet (`chanet`, no variant override)
- random initialization; no actor/critic/optimizer checkpoint
- 30 fresh 30-field instances per update: 4 complete episodes/rank × 4 GPUs
- 1,200 updates = **19,200 complete episodes**
- actor LR **1e-4**; critic LR **3e-3**; LR schedule none
- reward time, gamma 1.0, GAE 0.95, original PPO clip/epochs/batch
- entropy schedule computed on the full 1,200-update horizon
- candidate_attention=false; UAV randomize off; speed 8 m/s; coverage width 10 m
- checkpoint every 10 updates; greedy diagnostic/validation evaluation at
  update0 and every 20 updates
- `max_episode_steps=500` retained; smoke episodes reached at most ~60 steps,
  no truncation. If a future episode reaches 500, it is recorded as truncated,
  not completed.
- only validation panel selects best; diagnostic is observation only.

## Reference evaluations already frozen on the panels

Before training, on the same 30-field panels:

| panel/model | completed | mean time |
|---|---:|---:|
| NN diagnostic | 10/10 | 10,929.3 s |
| A10 u1060 diagnostic | 10/10 | 10,716.5 s |
| A10 u1200 diagnostic | 10/10 | 10,657.3 s |
| NN validation | 30/30 | 11,349.2 s |
| A10 u1060 validation | 30/30 | 11,059.3 s |
| A10 u1200 validation | 30/30 | 11,075.3 s |

All reference actions replay exactly (`replay_all_match=true`). These old
checkpoints were not reselected based on the 30-field panels.

## Tests

Relevant suites:

**57 passed** in
`test_random_instance_mode.py`, `test_random_instance_resume.py`,
`test_episode_rollout.py`, `test_chanet_ablation.py`,
`test_batched_encoder_equivalence.py`, `test_batched_policy_update.py`.

Added 30-field test covers fixed 30-field generation, distinct training
instances, and disjoint stream; existing tests cover panel isolation,
initialization, resume, full episodes and legacy 10-field A/B/C/D configs.

## Real 4-GPU smoke

Kept `num-updates=1200` and `eval-every=20`; used `stop-at=3`,
`checkpoint-every=1`, output `episode30/smoke30`.

- 4 ranks; 48 complete episodes; **0 failures, 0 truncations**
- 2,854 transitions; **16 optimizer steps/update**, 48 total for 3 updates
- 7 recorded geometry retries (only expected geometry errors); no episode failed
- full CHANet, random initialization, actual LR 1e-4/3e-3
- parameters updated; gates finite/pass
- all 48 instances have 30 fields; per-rank seeds/fingerprints unique
- diagnostic 10/10 and validation 30/30 at update0
- resume from `ckpt_update_3.pt` to update4 in independent output:
  field count, stream counter16, optimizer state, LR and gates preserved;
  no new initialization

### Measured cost and authorization

Smoke training accounting: **323.88 s for 3 updates = 107.96 s/update**.
1200 updates therefore require **≈35 h 59 m of training** before panel
evaluation. Update0 panel evaluation/startup adds roughly 5–6 minutes; 61
evaluation points at that scale imply several additional hours. The projected
total is roughly **40–42 hours**.

The user explicitly accepted this duration on 2026-11-07 and authorized the
formal run without reducing evaluation frequency, episodes or budget. The
formal launch SHA and actual first-20-update timing are appended below.

## Formal command (executed)

```bash
cd /home/ps/Desktop/Graph_planning/graph-uav-episode-10fields
EXP=/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode30
PYTHONPATH=. /home/ps/Desktop/Graph_planning/graph-uav-cpp/.venv/bin/python \
 -m coverage_env.rl.trainer.episode_random_worker \
 --diagnostic-bank $EXP/diagnostic10 \
 --validation-bank $EXP/validation30 \
 --output $EXP/formal_random_instances_s42 \
 --num-fields 30 \
 --num-updates 1200 --episodes-per-rank 4 --seed 42 \
 --instance-seed-base 1330000000 \
 --actor-lr 1e-4 --critic-lr 3e-3 --device cuda \
 --checkpoint-every 10 --eval-every 20
```

No sealed test, additional seed, or budget extension is authorized.

## Budget authorization and formal launch record

- Authorization: user accepted 40–42 h wall clock, removed 36 h cap.
- Launch SHA: **`dd4fac9c9c1108ae8358dc2dce4f00eed6cdcfce`**
- Actual start: **2026-11-07 21:02:30 +08:00**
- Persistent output: `episode30/formal_random_instances_s42/`
- No old 10-field actor/critic/optimizer is loaded; random initialization.
- No smoke checkpoint is resumed.

### Measured first 20 updates

- training accounting: **2,004.151 s / 20 updates = 100.21 s/update**
- projected training: **33.40 h** for 1,200 updates
- update0 evaluation: **280.55 s** (10 + 30 panels)
- update20 evaluation: **210.02 s**
- projected evaluation: **~3.58 h** for 61 points
- projected total: **~36.98 h**, finish approximately **2026-11-09 10:01 +08:00**
- at update20: **320 episodes**, **18,841 transitions**, **320 optimizer steps**
  (16/update); checkpoint size ~35 MB
- sampling + PPO optimization are jointly included in the 100.21 s/update;
  evaluation and checkpoint-save are not separately timed in the training log.
  Checkpoint writes are included in the update wall time.
- no failure/truncation/NaN or gate failure through update20.

The formal command in the next section was executed with the launch SHA above.
