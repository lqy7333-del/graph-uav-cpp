# M2 confirmatory payload-endpoint experiment

Frozen before execution. This is the independent follow-up to the M1 development screen.

## Design

- 20 new fixed 10-field maps: seeds **1,980,000–1,980,019**, formal area, distinct from
  M1 maps and all prior panels.
- Payload endpoints selected from the M1 development clue: **25 and 100** capacity units
  (0.5× and 2× of the original 50-unit payload). No further capacity values will be
  searched.
- Three frameworks: CC, D→P, P→D.
- 20 maps × 2 capacities × 3 methods = **120 searches**.
- Same map, energy capacity, cost/time model; search seed42; outer4/scored6, inner8/2,
  max 385 candidate evaluations or 1800 s; max_steps500.
- No RL training, no base relocation, no additional search seed/capacity/map.
- Capacity is applied to both max/remaining payload; D→P/P→D are rebuilt fresh at each
  capacity without reusing any partition, gate set, repair or cache from another capacity.

The two endpoints were chosen because of the M1 development observation. M1 is not part of
the M2 confirmatory sample; this selection is disclosed and not presented as preregistered
before M1.

## Primary tests

For map \(i\), capacity \(c\), method \(m\):

\[
d_{i,c,m}=T_{i,c,m}-T_{i,c,CC}.
\]

Primary interactions (low minus high):

\[
I_{i,D}=d_{i,25,D}-d_{i,100,D},\qquad
I_{i,P}=d_{i,25,P}-d_{i,100,P}.
\]

Positive \(I\) means the non-CC method is relatively worse at low capacity. The two
interactions are tested with an **instance/map-level paired bootstrap**, preserving all
three methods and both capacities on each resampled map; the two tests receive Holm
correction. Endpoint-specific differences, wins/ties/losses, resupplies, cost components,
wall-limit and candidate counts are auxiliary.

No result is filtered by whether it reaches 1800 s. Candidate counts and outer rounds are
reported. A common-candidate-budget analysis uses saved search curves only, without adding
runs or deleting wall-limited instances.

## Acceptance / stop

- Verify actual capacity, completion/return, and independent replay for all 120 cells.
- Preserve failures; do not replace maps or drop incomplete cells.
- Report search wall and candidate counts, including wall-limited runs.
- Stop after M2; no M3, no extra capacities/seeds/maps, no retuning after outcomes.
