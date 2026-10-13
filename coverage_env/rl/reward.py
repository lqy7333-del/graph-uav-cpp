"""Dimensionless, scale-normalized scalar reward for one Environment transition."""

from __future__ import annotations


def calculate_reward(
    result: dict,
    config: dict | None = None,
    *,
    distance_scale: float = 1.0,
    energy_scale: float = 1.0,
    coverage_scale: float = 1.0,
    fragmentation_scale: float = 1.0,
) -> float:
    """Return a reward with each physical term normalized to a common scale.

    The reward is ``-lambda1 * distance / distance_scale
    - lambda2 * energy / energy_scale + lambda3 * coverage / coverage_scale
    - lambda4 * fragmentation_penalty / fragmentation_scale``.

    Dividing each quantity by a problem-specific scale keeps the three terms
    comparable so no single term dominates the learning signal. Callers should
    pass a characteristic distance (meters), the UAV battery capacity (Wh), and
    the total coverable area (m²). Fragmentation is the positive increase in
    uncovered 8-neighbor components and small components caused by this step.
    Weights can be tuned in ``rl.reward_weights``.
    """
    config = config or {}
    weights = config.get("reward_weights", config.get("reward", {}))
    lambda1 = float(weights.get("lambda1", 1.0))
    lambda2 = float(weights.get("lambda2", 1.0))
    lambda3 = float(weights.get("lambda3", 1.0))
    lambda4 = float(weights.get("lambda4", 0.1))
    distance = result.get("distance", result.get("path_length", 0.0))
    energy = result.get("energy", result.get("energy_consumption", 0.0))
    coverage = result.get("coverage", result.get("coverage_gain", 0.0))
    fragmentation = result.get("fragmentation_penalty", 0.0)
    distance_term = float(distance) / max(float(distance_scale), 1e-9)
    energy_term = float(energy) / max(float(energy_scale), 1e-9)
    coverage_term = float(coverage) / max(float(coverage_scale), 1e-9)
    fragmentation_term = float(fragmentation) / max(float(fragmentation_scale), 1e-9)
    return (
        -lambda1 * distance_term
        -lambda2 * energy_term
        +lambda3 * coverage_term
        -lambda4 * fragmentation_term
    )
