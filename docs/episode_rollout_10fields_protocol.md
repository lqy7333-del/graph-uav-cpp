# Superseded protocol

The fixed-ten-instance training protocol in the earlier revision is **retracted**.
It was based on a misunderstanding of the experiment objective.

The current protocol is:

[random-instance fixed-10-field protocol](episode_rollout_random_10fields_protocol.md)

Summary:

- every training episode generates a fresh instance with exactly 10 fields;
- the old `train10` bank is a fixed **diagnostic panel**, not a training set;
- `valid30` is the fixed **validation panel**, not a training set;
- actor/critic start from random initialization;
- actor LR `1e-4`, critic LR `0.003`, 600 updates / 9,600 training episodes;
- formal training has **not** been started.

The legacy fixed-bank `episode_worker` remains available for historical
compatibility only and is not the formal protocol.
