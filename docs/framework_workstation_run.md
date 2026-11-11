# Framework workstation acceptance/formal run notes

- Base: `b018b89` (`codex/framework-baselines`); algorithm unchanged.
- Test-only adjustment: `test_framework_execution` compares the legacy oracle
  path and all numeric summary fields with a `1e-6`/`1e-8` floating tolerance.
  The only observed mismatch on this workstation was a ~1.4e-9 coordinate
  difference; regions, rewards, completion, resupply and scalar ledger values
  were otherwise identical. The same failure was reproduced at base `11b8fdf`.
- Acceptance panel: 3 new 10-field formal-area instances, seeds
  **1,900,400–1,900,402**, generated outside the formal 10/20 panels.
- Acceptance: full budget (outer4/6, inner8/2, 385 evaluations or 1800 s,
  seed42), 3 instances × 3 methods = 9 searches.
- Formal panels: the archived 10-field (`independent10`, seeds 1,500,000–29)
  and 20-field (`independent20`, seeds 1,800,000–29) snapshots. They are not
  regenerated. CC historical results are reused only after snapshot, physical
  config, budget, and replay checks; otherwise CC is rerun.
- Formal acceptance receipt must match the same code digest and physical
  config. No post-hoc changes to partitioning or search parameters.
- Wall-limit is a budget stop and retains a complete incumbent if one exists;
  it is not a task failure. Program/ledger/replay faults stop the queue.
- No RL training, 15/30 extensions, or sealed test.
