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
    reward_cfg = config.get("reward") if isinstance(config.get("reward"), dict) else {}
    mode = str(reward_cfg.get("mode", "distance")).lower()

    if mode == "time":
        # r_t = -(dT - dA_unique / (width * nominal_speed)) / T_scale
        #
        # The credit is the ideal straight-line time for the NEWLY covered area at a
        # fixed nominal speed, so it is identical for any policy that completes the
        # same task. Repeated coverage earns nothing (dA_unique is 0), and T_scale is
        # a single constant for the whole experiment -- never the instance area, the
        # field count, or a reference policy's cost.
        extra = float(result.get("time_total", result.get("time", 0.0)))
        new_area = float(result.get("coverage", 0.0))
        width = float(reward_cfg.get("coverage_width", 10.0))
        nominal = float(reward_cfg.get("nominal_work_speed", 8.0))
        time_scale = float(reward_cfg.get("time_scale", 100.0))
        if width <= 0.0 or nominal <= 0.0:
            raise ValueError("time reward needs positive coverage_width and nominal_work_speed")
        if time_scale <= 0.0:
            raise ValueError("time reward needs a positive time_scale")
        credit = new_area / (width * nominal)
        return -(extra - credit) / time_scale

    if mode != "distance":
        raise ValueError(f"unknown reward mode: {mode!r}")

    distance = float(result.get("distance", result.get("path_length", 0.0)))
    if total_area is None:
        total_area = float(config.get("total_area", 1.0))
    if not total_area or total_area <= 0.0:
        total_area = 1.0
    if scale is None:
        scale = float(config.get("reward_scale", 1.0))
    return -distance / float(total_area) * float(scale)
