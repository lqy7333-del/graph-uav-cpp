"""Total-path reward normalized by the episode's coverable area."""

from __future__ import annotations


def calculate_reward(
    result: dict,
    config: dict | None = None,
    *,
    total_area: float | None = None,
    scale: float | None = None,
) -> float:
    """Return ``-distance / total_area * scale`` for one transition.

    Minimizing total flight distance is the task objective, so the reward is
    negative and larger (closer to zero) for shorter paths. Dividing by the
    episode's total coverable area makes rewards comparable across episodes
    with different field counts and field sizes: the quantity is the traveled
    distance per unit area (the inverse of coverage efficiency).

    ``scale`` is a constant multiplier that only lifts the magnitude into a
    range PPO can learn from; it never depends on the instance, so it does not
    break the cross-instance normalization. Energy, coverage, and fragmentation
    no longer contribute to the reward.
    """
    config = config or {}
    distance = float(result.get("distance", result.get("path_length", 0.0)))
    if total_area is None:
        total_area = float(config.get("total_area", 1.0))
    if not total_area or total_area <= 0.0:
        total_area = 1.0
    if scale is None:
        scale = float(config.get("reward_scale", 1.0))
    return -distance / float(total_area) * float(scale)
