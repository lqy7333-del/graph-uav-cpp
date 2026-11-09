# Nested GA baseline: frozen implementation protocol

## Scope and provenance
Inspired by Xu et al., IEEE TSMC: Systems (2023), DOI 10.1109/TSMC.2022.3205695,
Section III-A. Their outer GA splits fields and inner GA merges subfields; their
subsequent assignment uses integer PSO. This implementation is an adaptation,
NOT a reproduction: outer repeated field visits, inner feasible entry/exit choices.
No field splitting, new scan angle, workload control, learned-model initialization,
or alternative resource/return rules are introduced.

## Representation and execution
Outer chromosomes are variable-length field-ID sequences allowing repeated visits,
bounded by max_steps (500). One-point splice crossover carries aligned inner keys;
swap/insert/delete/replace mutation supports repetition. Empty children are repaired
to one random field. Each step reads the actual action candidates and mask.
Unavailable/completed field slots are skipped. After exhaustion, a deterministic
nearest-centroid / nearest-entry action-space heuristic completes the task.
Its workload is computed by environment.step(CoverageAction), exactly as RL.
This initializer/completer is **not** the legacy NearestNeighborAgent, which also
computes its own requested workload. Report it separately from original NN.

Inner chromosomes have one real key per outer slot, interpreted as a quantile of
lexicographically sorted currently feasible entry/exit pairs for that field. -1
selects the fixed nearest-entry heuristic. Keys never authorize an infeasible action.
Uniform crossover and random-key mutation are used. Because gates change subsequent
states, every candidate is executed from an independent deepcopy of the same reset
instance. No reset/regeneration occurs during fitness evaluation.

Fitness is complete mission total_time including work, nonwork, maneuver, service,
automatic resupply and final return. A mission_complete reason AND returned_to_depot
are required. Incomplete candidates rank at infinity, not by their partial cost.
Programming exceptions propagate. Non-finite or non-closing time ledgers raise.
The exact best complete trace is retained globally and independently replayed without
repair; replay must complete and match time within 1e-6 s.

## Budget and genetic selection
Pilot defaults: outer population 8, outer generations 4; inner population 8,
inner generations 2; cap 500 simulator candidate evaluations or 300 search seconds.
Nearest initialization costs one evaluation. Every admitted outer candidate receives
exactly 16 evaluations, including elite reevaluations. Incomplete quotas do not enter
outer breeding; any complete solution they discover may update the global incumbent.
Outer breeding selects from the top half uniformly and retains the best parent.
Inner breeding does the same. Every evaluation (including failure/timeout) counts.
No cache is used. The budget includes environment copies, repairs, completion and
selection. Checks occur between simulation steps: a single simulator operation may
run past the deadline. Final independent replay is additional and timed separately.
Initialization failure raises rather than inventing a complete score.
Default maximum if generation limit alone were used: 1+8*4*8*2=513 evaluations;
the 500-evaluation cap therefore can interrupt the last inner quota.

## Pilot and outputs
Use 5 new 10-field development instances, 3 search seeds (42/43/44). Do not use
sealed/final evaluation to tune the GA. Pass the exact frozen experimental resolved
config explicitly; the runner does not quietly change field ranges, reward or UAV.
Generation failures propagate, not silently replaced. Each output directory must be
new. Save config, code SHA and working-tree status, full initial instance pickle and
its SHA256, all best actions, component summary, every evaluation's progress,
fallback/skipped counts, equal-quota records and replay summary.
Only load pickle snapshots from trusted experiment storage. The snapshot SHA is a
file integrity hash, not a canonical cross-version geometry fingerprint.

Example on workstation (replace CONFIG with the frozen evaluation config):

```bash
PYTHONPATH=. .venv/bin/python -m coverage_env.rl.evaluation.nested_ga \
  --config CONFIG --fields 10 --instances 5 --instance-seed 1400000 \
  --search-seeds 42 43 44 --max-evaluations 500 --wall-seconds 300 \
  --output /persistent/experiments/nested_ga/pilot10
```

Check seed manifests for collisions before pilot. Compare GA best with original NN,
action-space initializer and frozen CHANet on **the same saved instances**, and report
search wall time, model inference time and RL offline training separately. Inspect
fallback ratio before expanding to 20/30 fields. Report seed variability, complete
rate, incumbent-vs-time/evaluations and time components; do not label GA optimal.
No large experiment or training is authorized by this implementation commit alone.

## Implementation acceptance (2026-11-09)
- `pytest test_nested_ga.py test_turn_ledger.py test_maneuver_time.py`: 33 passed.
- New GA tests cover repeated visits, aligned mutation, deterministic search,
  equal inner quotas, capped partial quotas, incumbent retention, failures,
  incomplete missions, deadline, invalid settings, isolated state and exact replay.
- Real 5-field integration passed. CLI smoke uses the frozen 10-field resolved
  config wrapper, seed 1400100, search seed42, population2/2, generations1/1,
  max evaluations5, wall180. Five evaluations, 94 transitions, 49.60 search seconds,
  10.22 replay seconds. Incumbent 2721.698893 s equals initialization; no search
  improvement is claimed. Return completed, closure error 9.09e-13 s.
- See nested_ga_smoke.json. Production-sized pilot has not run.
