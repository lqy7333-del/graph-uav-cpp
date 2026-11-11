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

## Update after 47fdf97 — acceptance retest

The D→P coverage-selection fix was synced (`47fdf97`). Targeted suites:
**62 passed**.

The original failed instance was rerun from its frozen snapshot with the full
budget and independently replayed:

- instance `1900400`, D→P
- 385 evaluations, `evaluation_limit`, 6 complete outer rounds
- best cost **3184.68 s**
- replay cost **3184.68 s** (exact match), complete/returned

A fresh 9-search formal-area acceptance then completed all 3 instances × 3
methods with complete incumbents and replay verification. However the tool wrote:

```json
{"passed": false, "domain_verified": true, "search_exercised": false}
```

All nine task results are complete. The rejection is because P→D instance
`1900401` reached the 1800 s wall after 51 evaluations and **0 completed outer
rounds**; the acceptance gate requires at least one complete outer round for
every method/instance. P→D instance `1900400` had 1 complete outer round and
`1900402` had 1. The other methods had 1–6.

This is exactly the online-budget limitation the protocol says to report; it is
not a task failure. But `acceptance_receipt.json` is not a passing receipt, so
the formal queue is not allowed to start. No parameters, budgets, panel members
or receipt were altered to force a pass.

Fresh acceptance artifacts:
`/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework/acceptance10/run2/`

The formal 10/20-field queue remains **not started**.
