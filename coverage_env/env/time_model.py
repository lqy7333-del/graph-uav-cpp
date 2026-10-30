"""Angle-only maneuver-time proxy (a planning cost proxy, NOT a dynamics model).

    dt_turn(theta) = tau_pi * (|theta| / pi) ** p

* ``theta`` is the direction change between two adjacent non-zero straight segments.
* ``tau_pi`` is the reference extra time for one full reversal (180 degrees).
* ``p`` shapes the relative cost of small versus large maneuvers.

Both are declared simulation assumptions, not measured aircraft data, and the 3.0 s
reference below must not be described as a calibrated U-turn time.

Why angle-only
--------------
Because the earlier form also depended on the adjacent segment lengths, a junction
cost could not be reproduced from direction alone: splitting a 20 m inbound leg into
15 m + 5 m changed the clip. Removing the length dependence removes that failure mode.
The proxy is now invariant to duplicate points, collinear subdivision, and to how a
path is split across configuration calls.

It is NOT invariant to replacing one sharp corner with many small-angle waypoints;
that changes the geometry, so it legitimately changes the cost.

Billing point
-------------
When a new action fixes its first segment direction, the junction
"previous valid segment -> current first segment" is settled exactly once and charged
to the current step. A stop (resupply / takeoff) clears the validity flag, so a stop is
never billed as a continuous-flight turn.
"""
from __future__ import annotations

import math

DEFAULTS = {
    "tau_pi": 3.0,        # reference extra seconds for a 180-degree reversal
    "exponent": 2.0,      # 1 = linear, 2 = small angles almost free
    "cruise_speed": 8.0,  # base flight time = length / cruise_speed
    "service_time": 0.0,  # not modelled this round
}


def simplify(path):
    """Drop duplicate points and collinear interior points (no geometry change)."""
    points = []
    for p in path:
        point = (float(p[0]), float(p[1]))
        if not points or math.hypot(point[0] - points[-1][0],
                                    point[1] - points[-1][1]) > 1e-9:
            points.append(point)
    changed = True
    while changed and len(points) > 2:
        changed = False
        for i in range(1, len(points) - 1):
            a, b, cc = points[i - 1], points[i], points[i + 1]
            cross = (b[0] - a[0]) * (cc[1] - b[1]) - (b[1] - a[1]) * (cc[0] - b[0])
            dot = (b[0] - a[0]) * (cc[0] - b[0]) + (b[1] - a[1]) * (cc[1] - b[1])
            if abs(cross) <= 1e-9 and dot > 0:
                points.pop(i)
                changed = True
                break
    return points


def direction(seg_a, seg_b):
    """Unit direction of the segment a->b, or None when it has no length."""
    dx, dy = seg_b[0] - seg_a[0], seg_b[1] - seg_a[1]
    norm = math.hypot(dx, dy)
    return None if norm < 1e-9 else (dx / norm, dy / norm)


def turn_angle(prev_dir, next_dir):
    """Unsigned direction change in radians, in [0, pi]; 0 when either is missing."""
    if prev_dir is None or next_dir is None:
        return 0.0
    cross = prev_dir[0] * next_dir[1] - prev_dir[1] * next_dir[0]
    dot = prev_dir[0] * next_dir[0] + prev_dir[1] * next_dir[1]
    return abs(math.atan2(cross, dot))


def turn_cost(theta, **params):
    """Proxy extra seconds for one direction change of ``theta`` radians."""
    cfg = {**DEFAULTS, **params}
    if theta <= 0.0:
        return 0.0
    return cfg["tau_pi"] * (min(theta, math.pi) / math.pi) ** cfg["exponent"]


def path_time(prev_dir, path, **params):
    """(base_seconds, extra_seconds) for one flown path.

    ``extra`` charges exactly one junction cost at the path start (the incoming
    ``prev_dir``) plus one per interior vertex. Because endpoints are assumed to be at
    cruise speed and no length enters the cost, splitting a path at any existing
    vertex never changes the total.
    """
    cfg = {**DEFAULTS, **params}
    points = simplify(path)
    if len(points) < 2:
        return 0.0, 0.0
    base = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    base /= cfg["cruise_speed"]
    dirs = [direction(a, b) for a, b in zip(points, points[1:])]
    dirs = [d for d in dirs if d is not None]
    extra = turn_cost(turn_angle(prev_dir, dirs[0]), **params) if prev_dir and dirs else 0.0
    for a, b in zip(dirs, dirs[1:]):
        extra += turn_cost(turn_angle(a, b), **params)
    return base, extra


def heading_from_path(path):
    """Unit direction of the last non-zero segment of ``path`` (None if absent)."""
    points = [(float(p[0]), float(p[1])) for p in path or []]
    for a, b in zip(reversed(points[:-1]), reversed(points[1:])) if False else \
            list(zip(points[:-1], points[1:]))[::-1]:
        d = direction(a, b)
        if d is not None:
            return d
    return None
