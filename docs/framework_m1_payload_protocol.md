# M1 development response-screen: payload capacity

Frozen before execution. This is a development experiment, not M2 and not a sealed test.

## Scope

- 3 new independent, fixed **10-field** maps: seeds **1,970,000–1,970,002**, formal
  area 5,000–20,000 m², same workspace/energy/time/cost model.
- Payload multipliers: **0.5×, 1×, 2×** of each snapshot's original `max_payload`.
- Methods: CC, D→P, P→D.
- 27 searches = 3 maps × 3 capacities × 3 methods.
- Search seed42; outer4/scored6, inner8/2, max 385 evaluations or 1800 s, max_steps500.
- Global search-wall ceiling **13.5 h (48,600 s)**; stop after the current search if reached.
- No RL training, no weight/energy change, no base relocation, no M2, no additional seed.

## Capacity handling

For each search, deep-load the frozen snapshot and set both `max_payload` and
`remaining_payload` to `original_max_payload × multiplier`. `payload_per_area` and all
other physical parameters remain fixed. D→P/P→D are instantiated fresh at that capacity:
no task partition, gate set, repair, or decode cache from another capacity is reused.
CC also starts fresh.

## Acceptance

- All 27 cells attempted unless the global wall cap is reached.
- Report actual `max_payload` and snapshot hash per cell.
- Replay every complete incumbent with the original actions; preserve failures.
- Capacity is verified from the run state, not inferred from the factor.
- No method-win criterion. Analysis is a framework × capacity response screen on only
  three maps, with no population-level significance claim.

## Endpoints

Primary: same-map change in `D/P − CC` total-time difference across capacities (interaction),
reported cell-by-cell and as descriptive mean differences. Direction reversal is a possible
outcome, not a failure.

Mechanism: pre-resupply payload/energy, departures/resupplies, return and non-return
nonwork/time, maneuver, scan segments/internal connections. If the primary difference
changes without corresponding mechanism changes, only relative-performance response is
supported. If no discernible interaction is seen, report no interaction detected in this
small screen, not absence of a mechanism.
