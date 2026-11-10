# Nested-GA independent scale comparison: 15 and 20 fields

**Frozen before any 15/20-field policy evaluation.**  
No RL training, no GA parameter changes, no 30-field model, no 30-field extension.

## Provenance

- Baseline branch: `codex/nested-ga-baseline`
- Base SHA: `947f32b4d63449affeb82e94dbcc2db216add65d`
- 10-field result: archived at
  `docs/nested_ga_independent10_seed42/`; it is referenced, **not rerun**, and is
  not represented as part of a single simultaneous preregistration.
- Code: frozen nested GA from `136074e` plus logging/queue fixes and the
  same-snapshot control adapter. No GA search-rule changes.
- GA initialization is never RL-based.

## Panels and environment

- 15 fields: 30 new instances, candidate seed block **1,700,000**
- 20 fields: 30 new instances, candidate seed block **1,800,000**
- Same area 5,000–20,000 m², workspace, shape/spacing, UAV capability, time
  model, workload, resource/resupply and return/completion rules as A10.
- Fixed-order generation; only expected geometry errors may advance the seed.
  Every rejection/replacement is recorded. No policy-based map selection.
- Full snapshots, config and SHA256 frozen before any policy evaluation.
- Historical manifests/training streams checked for exact seed overlap before
  any strategy result; collision, if any, would force a new full block.
- All methods read the same frozen snapshot for a given instance.

## GA method and budget

Method label:
**“嵌套GA，每次最多385次候选评估或1800秒搜索时间，先到者停止。”**

Per scale:

- 30 instances × search seed **42** only = 30 searches
- Outer population **4**, at most **6 scored rounds**
- Inner population **8**, **2** generations
- Planned evaluations **1 + 4×6×16 = 385**
- Wall safety cap **1800 s**
- max_steps 500
- Same encoding, crossover, mutation, selection, completion, fitness.
- No RL initialization.
- Total two-scale cap: **60 × 1800 s = 30 h** search wall, plus generation,
  controls, replay and reporting. Authorized.

A normal wall-limit is a budget stop, not a task failure. It retains the best
complete incumbent and enters the formal statistics. Program exceptions,
non-finite values, ledger failures, replay mismatches, or no feasible complete
incumbent stop the queue with diagnostics. No deletion of failures or
success-only means.

## Controls on each same snapshot

1. original NN;
2. GA action-space nearest initializer;
3. frozen A10 CHANet u1060;
4. frozen A10 CHANet u1200.

A10 checkpoints are fixed:
- u1060 `82d67f410750cc525e1a9d23a9a542bea3bdd51ce789669cd16067818d70bc14`
- u1200 `10adbffca211cbb711793d3b2d19182a0ec66329cf40151ce8523185d0c63ab4`

CHANet trained only on 10-field maps; 15/20-field results are cross-scale
inference. GA searches each new instance online. The 30-field trained model is
not included. NN and GA initializer remain separate baselines.

Before the formal panels, run one 15-field and one 20-field small-budget smoke
outside the panels: at most 5 evaluations, verify field count, completion,
return, ledger closure and independent action replay; record per-candidate
runtime. Any correctness failure stops and reports.

## Statistics

Reported **separately for 15 and 20 fields**; no pooled scale mean.

For each scale, instance is the paired unit:

- `NN − GA`: positive means GA faster.
- `GA − A10 u1060`: positive means CHANet faster.

Report mean and median paired difference, wins/ties/losses, instance-level paired
bootstrap 95% CI, and relative mean cost difference.

There are **four primary tests** (two scales × two comparisons). If p-values are
reported, Holm-adjust the four tests. Keep unadjusted effect intervals and label
them as such. These are single-GA-seed, single-RL-seed development comparisons;
no cross-seed stability claim.

Auxiliary:
- `A10_u1060 − NN` paired effect;
- GA vs initializer;
- GA vs A10 u1200;
- nonwork+maneuver+service (deducted work) for each;
- total/work/nonwork/maneuver/service, distance, visits, resupplies, coverage,
  completion.

## Compute reporting

Per scale report candidate evaluations/transitions, completed outer rounds,
full/partial inner quotas, wall-limit rate, fallback ratio and source,
incumbent-refresh count, search/replay/total wall. Report NN/CHANet online
episode wall separately, including environment execution and excluding model
loading; report offline RL training separately. Do not compare a single model
forward with a complete GA search.

If many large-scale runs cannot finish the initial outer scoring, state that this
is an online-budget limitation of the current implementation and not an estimate
of GA performance with a larger budget.

## Stop conditions

After all 60 searches and controls: summarize, commit/push and stop. Do not add
search seeds, increase budget, train RL, or start 30 fields.
