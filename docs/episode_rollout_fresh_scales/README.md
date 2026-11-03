# Frozen u1060: fresh 10/20/30-field paired greedy evaluation

- Frozen checkpoint: `formal_random_instances_s42/ckpt_update_1060.pt`
  (validation-selected; SHA256 `82d67f41…`)
- No training, no model selection, no new tuning.
- 20 fresh instances per scale; same instance is used for the frozen policy and
  NN, with the same seed.
- All 60 policy and 60 NN episodes completed.
- Time is the **model-estimated total greedy time**.

## Results

| fields | policy mean (s) | NN mean (s) | NN − policy (s) | 95% paired CI | policy wins |
|---:|---:|---:|---:|---:|---:|
| 10 | **3241.2** | 3371.5 | **+130.3** | [+96.3, +164.7] | 19/20 |
| 20 | **6553.7** | 6706.8 | **+153.1** | [+88.4, +217.7] | 16/20 |
| 30 | **10743.5** | 10941.2 | **+197.7** | [+139.6, +249.8] | 19/20 |

Positive `NN − policy` means the frozen policy is faster. Completion was 20/20
for both methods at every scale. The paired differences are conditional on the
20 fresh instances at each scale; they are not a cross-scale generalization
proof.

## New seed sets

- 10 fields: `1,000,000–1,000,019` (20 accepted)
- 20 fields: `1,010,000–1,010,019` (20 accepted)
- 30 fields: `1,020,000–1,020,020` excluding `1,020,017` (20 accepted; one
  geometry-placement rejection, recorded only by generation validity)

Seeds were screened only for successful geometry generation, in fixed order,
never by policy performance. The 30-field rejection is preserved in
`summary.json`. All instances use the same area range (5,000–20,000 m²) and
generator configuration as the frozen checkpoint's protocol.

## Files

- `summary.json` — per-scale means, CIs, wins, seeds/fingerprints,
  generation-rejection log
- `paired_rows.csv` — one row per policy/NN episode
- Persistent `instances.pt` — full snapshots:
  `/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/fresh_scales_u1060/instances.pt`

These 60 instances are a fresh evaluation panel, not the earlier fixed
diagnostic/validation panels and not the unused final sealed test set.
