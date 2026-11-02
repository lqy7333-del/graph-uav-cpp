# Fixed-instance overfit: recovery check and persistent storage rules

## 1. Bounded recovery check (2026-11-02)

Searched `/home/ps`, `/mnt/sda2`, `/media/ps`, Trash, recent `.pt` files and
the Git worktrees. Findings:

- Original `run_seed42/ckpt_update_780.pt`: **not found**.
- Original fixed bank (`instances.pt` + manifest): **not found**.
- `best.json`: only a summary; it is not a model/checkpoint file.
- `overfit_best*.pt` in the VS Code trash are older unrelated checkpoints
  (`encoder/decoder`, no actor/critic/rank_runtime); not usable.
- Remote branch has reports/CSV/figures, but no weights or bank.
- **B42/240 source checkpoint is present**:
  `/home/ps/Desktop/Graph_planning/graph-uav-cpp/coverage_env/results/maneuver_time/formal_fixed_gate_v2/B_seed42/ckpt_update_240.pt`
  SHA256 `71cfda4f20805e553888854ebd12ba47b6244fbe25ce7ae0a62f7cdf8faa7560`.

Conclusion: the original 780 training state cannot be restored. Do not call a
rebuilt model the same continuation.

## 2. Persistent root

All new artifacts go to:

`/home/ps/Desktop/Graph_planning/graph-uav-experiments/overfit20/`

On the 916 GB root filesystem (not `/tmp`), with at least 690 GB free at start.

- `initial/` — B42/240 copy + SHA256SUMS
- `bank/` — instances.pt, manifest.json
- `new780/` — new common start trained from B42/240 through update 780
- `control_lr3e4/`, `experiment_lr3e5/` — paired branches from new780
- `backups/` — independent copies of the common start and each endpoint
- logs/checksums at every stage

Git keeps code and protocol; weights and bank are not committed.

## 3. New bank identity

Regenerated with the original code/config/checkpoint and seeds 70000–70009.
The new `instances.pt` SHA256 begins `48382c40d40a71ac…`, matching the original
recorded bank hash prefix; the manifest SHA differs because the initialization
path is now persistent. The bank is registered as a **new manifest** even though
the instance content hash prefix matches.

## 4. New common start

The original 780 is unavailable, so the common start is a **new 780** trained
from the same B42/240 with the same bank, target budget 1000, and the same
overfit settings. It is explicitly **not** the original 780 and may have a
different score. Selection of the branch point is pre-specified by update
number, not by choosing the best curve point.

## 5. Branch experiment

From new780:

- control: actor LR 3e-4, 781–1000;
- experiment: actor LR 3e-5, 781–1000;
- both restore optimizer state, rank runtime, RNG, environment and entropy
  schedule; critic/reward/network/bank/rollout/decode unchanged;
- frozen new780 is the third reference;
- primary decision uses the fixed endpoint 1000, not the minimum checkpoint.

Only after the new780 exists will a backup be made and a from-backup restore
test run. The same is required at both branch endpoints.

## 6. Status

The long new-780 rebuild was stopped and archived; it is not used. The accepted
screen is the 60-update pair from B42/240 (`short/control_s42`,
`short/lr3e5_s42`). Results: `docs/fixed_instance_overfit_short_screen.md`.
