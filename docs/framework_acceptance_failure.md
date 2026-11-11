# Framework formal-area acceptance failure

Date: 2026-11-11  
Branch: `codex/framework-baselines`  
Base commit: `b018b89`  
Run commit: `4fdb585` (test-only float tolerance + run notes)  
Status: **STOPPED before formal panel.**

## What ran

- Targeted tests: **61 passed** after changing only a cross-platform float
  comparison in `test_framework_execution.py`; the same mismatch reproduced at
  base `11b8fdf` and was a ~1.4e-9 coordinate/scalar difference, not a different
  execution result.
- Formal-area acceptance panel: 3 new 10-field instances,
  seeds **1,900,400–1,900,402**, area 5,000–20,000 m².
- Acceptance command:
  `framework_comparison --stage acceptance --max-evaluations 385 --wall-seconds 1800 --seed 42`
- CC completed one full search and its result was saved.
- D→P failed on instance `1900400` during candidate execution.
- P→D was not started.

## Failure

```
coverage_env/baseline/divide_then_plan.py:158 in _evaluate_decoded
  env.step_region(...)
coverage_env/planner/framework_execution.py:78
  plan = executor.plan_region(...)
coverage_env/planner/configuration_executor.py:241 in validate_plan
ValueError: Planned strokes do not cover assigned grid points
```

This is a D→P regional-planning/coverage-validation error, not a wall-time or
task-completion result. It blocks the acceptance receipt.

## Artifacts preserved

`/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework/acceptance10/run/`

- `0000_cc.json` — complete CC search result
- `failure.json` — exact identity and exception
- `run_contract.json` — frozen manifest/code/config contract
- `progress.json` — 1/9 completed searches
- no `acceptance_receipt.json`

Acceptance manifest and snapshots:
`/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework/acceptance10/`

## Stop decision

Per protocol, because the formal-area acceptance did not complete and D→P raised
an implementation error, no formal 10/20-field framework queue was started. No
search parameters, regions, budgets, or instances were changed to force a pass.
The failure is left for code correction and a new acceptance run.
