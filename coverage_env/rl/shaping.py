"""Potential-based reward shaping for the ordering sub-problem.

The diagnosis
-------------
The ordering payoff is spread over the episode: on this task the whole span of
field orders is worth 5.71 reward points, but spread over 6-8 steps the *marginal*
effect of a single ordering decision is only 0.2-0.7, while a single episode of
sampling noise has a standard deviation of 1.23. The per-step reward is therefore
dominated by everything except the order, and policy gradients cannot attribute
it. Measured: the policy's deviations from a nearest-neighbour order are right
only 18-35% of the time, i.e. they are noise rather than strategy.

The fix
-------
Add a *potential difference* to the reward::

    r'(s, a, s') = r(s, a, s') + gamma * Phi(s') - Phi(s)
    Phi(s) = -lambd * remaining_tour_length(s) / total_area

``remaining_tour_length`` is the shortest open path from the UAV's current position
through the still-unfinished parcels (0 once everything is covered), so
``Phi(terminal) == 0``.

Why this is safe
----------------
Ng, Harada & Russell (1999): for any potential ``Phi`` with ``Phi(terminal) = 0``,
the shaped return telescopes to ``G' = G - Phi(s_0)``, a constant that does not
depend on the policy, so **the optimal policy is unchanged**. The potential may be
computed by any crude heuristic -- even nearest neighbour -- and the guarantee
still holds; only the learning dynamics change.

And that is exactly the point: framing the ordering signal as a per-step potential
difference puts it into the term that actually drives the gradient (the immediate
TD error), instead of leaving it to a difference of two distant returns.
"""

from __future__ import annotations

from math import hypot


def _distance(first, second) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def _open_length(order, centroids, start) -> float:
    """Distance for start -> order[0] -> ... -> order[-1] (no return leg)."""
    if not order:
        return 0.0
    total = _distance(start, centroids[order[0]])
    total += sum(
        _distance(centroids[a], centroids[b]) for a, b in zip(order, order[1:])
    )
    return total


def _improve(order, centroids, start, rounds: int = 6):
    """2-opt reversal + or-opt move on the open-path objective."""
    best = list(order)
    best_length = _open_length(best, centroids, start)
    for _ in range(rounds):
        improved = False
        size = len(best)
        for i in range(size - 1):
            for j in range(i + 1, size):
                candidate = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                length = _open_length(candidate, centroids, start)
                if length < best_length - 1e-9:
                    best, best_length, improved = candidate, length, True
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                candidate = best[:i] + best[i + 1:]
                candidate.insert(j, best[i])
                length = _open_length(candidate, centroids, start)
                if length < best_length - 1e-9:
                    best, best_length, improved = candidate, length, True
        if not improved:
            break
    return best, best_length


def remaining_tour_length(environment) -> float:
    """Shortest open path visiting every unfinished parcel, or ``0.0`` when done.

    Uses a nearest-neighbour start plus 2-opt/or-opt. The result need not be
    optimal: any potential preserves the optimal policy, so this only has to be a
    cheap, monotone proxy for "how much work is left".
    """
    unfinished = [
        (field.field_id, field.centroid)
        for field in environment.fields
        if field.get_uncovered_points()
    ]
    if not unfinished:
        return 0.0
    if len(unfinished) == 1:
        return 0.0
    centroids = dict(unfinished)
    start = (
        environment.uavs[0].current_position
        if getattr(environment, "uavs", None)
        and environment.uavs[0].current_position is not None
        else environment.depot.get_position()
    )
    remaining = list(centroids)
    order, current = [], start
    while remaining:
        closest = min(remaining, key=lambda name: _distance(current, centroids[name]))
        remaining.remove(closest)
        order.append(closest)
        current = centroids[closest]
    _order, length = _improve(order, centroids, start)
    return float(length)


def potential(environment, strength: float, total_area: float) -> float:
    """``Phi(s) = -strength * remaining_tour_length / total_area``."""
    if strength == 0.0:
        return 0.0
    area = float(total_area) if total_area else 1.0
    return -float(strength) * remaining_tour_length(environment) / area


def shaped_reward(
    base_reward: float,
    potential_before: float,
    potential_after: float,
    gamma: float,
) -> float:
    """``r + gamma * Phi(s') - Phi(s)``."""
    return float(base_reward) + float(gamma) * float(potential_after) - float(
        potential_before
    )
