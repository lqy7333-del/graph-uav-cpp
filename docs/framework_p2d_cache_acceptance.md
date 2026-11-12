# P→D deterministic decode cache — acceptance

Date: 2026-11-12. Base fix: `47fdf97`; cache implementation: `15b3bfd`.
No search parameters or budgets changed.

## Profiling on 1900401 (bounded, no 1800 s search)

One fixed best chromosome/gates from the prior accepted run:

- contextual division: **0.00055 s**
- connected-component work: **0.00124 s** (16 calls)
- gate planning/validation: **16.92 s** (469 gate validations; validation itself 3.68 s)
- other repair bookkeeping: ~0.006 s
- partition total: **16.93 s**
- one complete execution: **2.38 s**
- 210 initial feasible-gate probes for this chromosome

The repeated per-inner-candidate region/gate work, not task execution, was the
dominant cost.

## Cache definition

Within one inner quota, for each unchanged outer visit/weight chromosome, the
implementation now caches only:

- discrete contextual regions and ownership;
- connected-subregion repair/task list;
- full-resource feasible gate sets.

It does **not** cache environment state, current position, remaining
resources, resupply decisions, selected gates across inner candidates, paths
from earlier candidates, or cost. Every inner candidate still deep-copies the
initial environment and executes a complete task.

The cache is cleared at every new outer candidate (`inner` entry). The first
decode remains inside the wall clock; hit count, hit preprocess time, saved work
and saved probes are recorded.

## Equivalence

Fixed snapshot `1900401`, fixed chromosome and gates:

- region ownership, selected tasks, repair records: identical
- action trace: identical after JSON canonicalization
- completion and all time-ledger components: identical
- total cost: **3768.180015186167 s** in both miss and hit
- one full inner quota (16 evaluations): **15 hits, 1 miss**, cost 3768.18
- first decode 17.21 s; cache-hit decode 0.00072 s; 360/25 hits on the first
  full P→D acceptance search.

## Acceptance result

New independent acceptance directory `acceptance10/run3`:

```json
{"passed": true, "domain_verified": true, "search_exercised": true}
```

All 9 searches completed and replayed. P→D reached 385 evaluations / 6 outer
rounds on two instances and 351 evaluations / 5 rounds on the wall-limited
instance; the prior 0-round failure is removed at unchanged budget.

Formal queue release condition: acceptance passed; estimated new search wall
(CC reused, D→P/P→D on 60 instances, worst case 120×1800 s = 60 h) is below the
90 h ceiling, before replay/overhead.
