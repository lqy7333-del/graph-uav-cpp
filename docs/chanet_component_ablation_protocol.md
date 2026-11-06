# CHANet component ablations C/D (paired seed42)

Status: implementation/smoke complete; formal training started after commit freeze.

- Baseline A: `formal_random_instances_s42` (full CHANet)
- Reference for paired initialization: A `ckpt_update_0.pt`
  SHA256 `114d11190d1282edc9377db2f846244a1420792a9cf11542ee3553189f7443db`
- No A/B rerun. No new seeds or budget.

## Definitions

- **C = no cross-field boundary interaction** (`no_cross_field_boundary`):
  replaces `BoundaryInterEncoder` (the `EDGE_BB_INTER` layer). It does not read
  cross-field neighbours or their edge attributes.
- **D = no intra-boundary interaction** (`no_intra_boundary`):
  replaces `BoundaryIntraEncoder` (the `EDGE_BB_INTRA` layer). It does not read
  same-field boundary neighbours or their edge attributes.
- Both retain BoundaryIntra/Inter counterparts, `FieldBoundaryEncoder`,
  `UavRelativeEncoder`, global pooling, raw features, action embeddings and the
  original hierarchical decoder. C is not “remove all global information”; D is
  not “remove all intra-field information.”

## Replacement module and parameter fairness

`NodeLocalResidual` is a genuine residual per-node transform:

`LayerNorm(x + Linear(GELU(Linear(x))))`

- no neighbour aggregation, no edge/attributes;
- input/output hidden width 128 unchanged;
- residual, LayerNorm, GELU retained; dropout inherited;
- hidden width chosen only by the replaced `RelationGATLayer` parameter count;
  width **902** at H128/heads4;
- replaced module: **232,198** params vs original **232,320** (−122);
- encoder: **983,174** vs A **983,296** (−122, −0.0124%);
- actor: **1,252,105** vs A **1,252,227** (−122).

No replaced-module output is zeroed and no dead parameters are retained. The
C and D replacements have the same parameter count by construction.

## Paired initialization

`--paired-init` is A update0 only. The loader:

1. rejects nonzero source update, trained initialization, wrong encoder/config;
2. copies all target tensors whose modules were not replaced, including
   `encoder.field_boundary_encoder`, `encoder.uav_relative_encoder`, the
   retained boundary layer, feature embedding, global encoder, action embedding,
   decoder and critic;
3. leaves only the replaced boundary module randomly initialized;
4. creates new actor/critic optimizers;
5. uses strict state loading and records reference SHA, copied tensor count,
   replaced modules and parameter counts.

Smoke metadata: C and D each copied **85 actor tensors**, source update 0,
encoder variant and `replaced_modules` recorded.

## Training conditions

Same as A seed42:

- random 10-field instance at every training episode, seed base 900,000,000;
- 4 episodes/rank, 4 GPUs = 16/update;
- 1,200 updates = 19,200 episodes;
- actor LR 1e-4; critic LR 3e-3; LR schedule none;
- time reward, gamma1, GAE0.95, same entropy schedule;
- candidate_attention false, fixed UAV, original decoder;
- checkpoint every 10 updates; full diagnostic/validation greedy evaluation
  every 20 updates and update0;
- `train10` diagnostic only; `valid30` selects model;
- no tuning during training, no extra seeds, no sealed test.

Only encoder variant/replacement metadata differs from A. The decoder, reward,
PPO hyperparameters and instance stream are unchanged.

## Smoke results

Both C and D with A update0, `num-updates=1200`, `stop-at=3`,
`checkpoint-every=1`, `eval-every=20`:

| item | C | D |
|---|---:|---:|
| updates run | 3 | 3 |
| episodes/rank | 12 | 12 |
| transitions (4 ranks) | 975 | 961 |
| synchronized optimizer steps | 24 | 24 |
| training accounting | 82.1 s | 80.1 s |
| wall incl. update0 panels | ~290 s | ~309 s |
| failures/truncations | 0 | 0 |
| gates/NaN/DDP errors | none | none |
| variant | `no_cross_field_boundary` | `no_intra_boundary` |
| actor/critic LR | 1e-4 / 3e-3 | 1e-4 / 3e-3 |
| first 12 seed/fingerprints | match A | match A |

Resume from each `ckpt_update_3.pt` to update4 in an independent output, without
`--paired-init`: variant and paired provenance preserved, stream counter 16,
4 episodes/rank, seeds/fingerprints match A rows 13–16, parameters updated,
gates passed.

## Formal commands

C:

```bash
cd /home/ps/Desktop/Graph_planning/graph-uav-episode-10fields
PYTHONPATH=. /home/ps/Desktop/Graph_planning/graph-uav-cpp/.venv/bin/python \
  -m coverage_env.rl.trainer.episode_random_worker \
  --diagnostic-bank /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/train10 \
  --validation-bank /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/valid30 \
  --output /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/ablation_no_inter_s42 \
  --encoder-variant no_cross_field_boundary \
  --paired-init /home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_instances_s42/ckpt_update_0.pt \
  --num-updates 1200 --episodes-per-rank 4 --seed 42 \
  --instance-seed-base 900000000 --actor-lr 1e-4 --critic-lr 3e-3 \
  --device cuda --checkpoint-every 10 --eval-every 20
```

D is identical except:

```text
--output .../ablation_no_intra_s42
--encoder-variant no_intra_boundary
```

## Planned analysis

- A_u1060 vs C/D validation-selected best; A_u1200 vs C/D_u1200.
- Difference C/D − A; **positive means A better**.
- Raw diagnostic/validation curves with NN; completion; components; per-instance
  paired differences and win/loss; parameters, inference and wall time; sample
  volume and instance-flow/gate records.
- Do not sum C and D differences as an additive CHANet contribution.
- Single paired training seed; no sealed test.
