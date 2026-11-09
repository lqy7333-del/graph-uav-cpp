# Independent 10-field nested-GA comparison — frozen pre-run protocol

**Status: frozen before any independent-panel policy evaluation.**  
No RL training, no GA parameter search, no 20/30-field run.

## Provenance and audit

- Algorithm/base: `136074e` (nested GA)
- Recording fixes: `d10628b`
- Corrected pilot/docs: `0537100`
- The GA algorithm (encoding, crossover, mutation, selection, completion,
  fitness) is unchanged.
- Corrected pilot audit: 5 runs contain **24 incumbent refresh events**:
  **5 initialization events** (one per run) and **19 search improvements**.
  The first history row of every run is the initialization incumbent. The prior
  “22 / only 1 initialization” wording was wrong.
- Old 8×4 pilot remains a development record: it did not reliably finish the
  initial outer population and its old `inner_quotas` omitted interrupted
  quotas. It is not used for the independent comparison.

## Panel

- 10 fields per instance; area 5,000–20,000 m²
- Same workspace, geometry, spacing, UAV, coverage, workload, time,
  resupply/return and completion rules as the 10-field formal run
- Instance seed block starts at **1,500,000**, generating in fixed order until
  30 valid instances exist; only known geometry generation errors may be
  replaced, and every failure/replacement is recorded
- Exact collision check against all known manifests/training streams before
  evaluating any policy
- Full snapshots, manifest, config and SHA256 saved before evaluation
- This is a new 30-instance panel; the old 5 pilot panel is not reused

## Frozen GA budget

- Per instance independent search seeds: **42, 43, 44**
- Instances: **30** → **90 searches**
- Outer population **4**; scored outer rounds **6**
- Inner population **8**; inner generations **2**
- Planned evaluations per search: **1 + 4×6×16 = 385**
- `max_steps=500`
- Safety wall per search: **1,800 s**, frozen before running
- No RL route initialization; no model access
- If wall reached: retain best complete incumbent, mark incomplete budget, do not
  delete/replace/rerun for a better score. Three consecutive wall hits pause the
  queue for cost inspection.
- Total preparation: approximately 18–24 h; ordinary runtime variation does not
  shrink instances/seeds.

## Controls on the exact same snapshots

1. original NearestNeighborAgent (legacy workload);
2. GA action-space nearest-field/entry initializer (separate baseline);
3. frozen CHANet A10 u1060;
4. frozen CHANet A10 u1200.

GA is never initialized from CHANet or NN weights/actions. All methods read the
same pickle snapshot; every action trace is replay-verified.

## Primary comparisons and sign

Unit of analysis is the **instance** (30), not the 90 searches.

- `NN − GA_mean`: positive means GA faster.
- `GA_mean − CHANet_u1060`: positive means CHANet faster.
- For each: mean/median paired difference, wins/ties/losses, instance-level
  paired bootstrap 95% CI, relative improvement.
- Two primary comparisons are one family: report unadjusted bootstrap intervals
  and Holm-adjusted p-values. Three GA seeds are retained within each instance
  for the instance-level bootstrap.
- Auxiliary: GA vs initializer, GA vs CHANet u1200, seed sd/min/max, best-of-3
  (marked 3× budget), deducted cost = nonwork+maneuver+service.
- Completion is checked first; success-subset means cannot replace full-task
  cost.

## Reports and boundaries

Record per search: evaluations, full/partial inner quotas, outer rounds/children,
fallback sources and counts, refresh events, failed candidates, transitions,
search/replay/total wall and raw incumbent trajectories. Record control online
episode wall separately from model loading and RL offline training.

This is an adapted nested-GA baseline, not a direct reproduction of the cited
paper. CHANet has one training seed; GA has three search seeds with different
meaning. This is not a sealed test.

## Revision 1 — resume after first wall-limit pause

**Timestamp: 2026-11-09 22:40 +08:00. Completed searches at revision: 6 / 90.**
Revision basis: observed computation-cost heterogeneity and a correctness check;
not model win/loss.

- The first three consecutive wall-limited searches occurred on instance
  1,500,001. The queue was paused once for inspection, as originally specified.
- After inspection, **all subsequent wall-limit outcomes are normal recorded
  results and do not pause the queue**. Every valid best complete incumbent is
  retained and included in the final statistics.
- A normal wall-limit is not a program failure.
- The queue still stops and preserves diagnostics for: program exceptions,
  non-finite values, ledger closure failure, replay mismatch, or absence of a
  feasible complete incumbent.
- The six already completed searches (three evaluation-limited at 1,500,000,
  three wall-limited at 1,500,001) are retained unchanged, not rerun, and not
  given extra budget.
- No total 18–24 h cap is applied. The remaining 84 searches may consume up to
  approximately 42 h of search wall time, plus replay/control/postprocessing.
- The frozen method description is:
  **“嵌套GA，每次最多385次候选评估或1800秒搜索时间，先到者停止。”**
- We do not claim that every run completed six outer rounds, and we do not claim
  search convergence.
- Three separate reporting fields will be used:
  1. task-plan completion: incumbent completed and returned;
  2. planned-search completion: six outer rounds and 24 full inner quotas;
  3. stop reason: evaluation limit, wall limit, or exception.
  A wall-limited run with a complete best incumbent is not a failed path.
- All 30 instances and all 90 searches enter the final statistics. Per-instance
  GA cost is the mean of three best complete incumbent costs; no completed-only
  filtering is allowed.

## Revision 2 — single search seed (2026-11-09 23:28 +08:00)

**Completed at revision: 3 seed42 results; 5 historical seed43/44 results.**
One in-flight seed44 search was safely terminated; no result file was written.

- User reduced the evaluation budget to **30 instances × seed42 only**.
- Seed42 was the predetermined primary search seed, not selected by observed
  performance.
- All completed seed42 results are retained unchanged; no rerun and no extra
  budget.
- Completed seed43/44 results are moved to `historical_seed43_44/` as historical
  extras. They do not enter means, best-of, or any main statistic.
- No three-seed mean, seed-standard-deviation, or best-of-3 analysis is reported.
- Controls (NN, GA initializer, CHANet u1060/u1200) are reused unchanged on the
  same 30 snapshots.
- Search budget remains **385 evaluations or 1800 s, first reached stops**.
- Main statistics: 30 instances, one GA search seed, one RL training seed.
