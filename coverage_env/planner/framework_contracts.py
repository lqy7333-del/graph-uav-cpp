"""Pure allocation contracts for framework baselines (not an execution API).

Point IDs are scoped by field and refer to the frozen instance grid. These
checks prove discrete task ownership, NOT swept geometry or resource feasibility.
No environment, GA, or trained policy imports this module yet.
"""
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from math import isfinite


@dataclass(frozen=True)
class RegionTask:
    """A region assigned before path execution, with immutable point ownership."""

    task_id: str
    field_id: str | int
    point_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a nonempty string")
        if isinstance(self.field_id, bool) or not isinstance(self.field_id, (str, int)):
            raise ValueError("field_id must be a string or integer")
        if not isinstance(self.point_ids, tuple) or not self.point_ids:
            raise ValueError("point_ids must be a nonempty tuple")
        if any(not isinstance(p, str) or not p for p in self.point_ids):
            raise ValueError("point IDs must be nonempty strings")
        if len(set(self.point_ids)) != len(self.point_ids):
            raise ValueError("duplicate point within region")


def validate_partition(
    tasks: Sequence[RegionTask],
    expected: Mapping[str | int, Sequence[str]],
) -> None:
    """Require exact, disjoint ownership of every expected discrete point.

The caller supplies the frozen grid (or explicitly declared remaining grid).
No missing point is silently assigned to a fallback region.
"""
    if not expected or any(not points for points in expected.values()):
        raise ValueError("expected grid must contain nonempty fields")
    required = set()
    for field, points in expected.items():
        if len(set(points)) != len(points):
            raise ValueError("duplicate point in expected grid")
        required.update((field, p) for p in points)
    seen = set()
    task_ids = set()
    for task in tasks:
        if task.task_id in task_ids:
            raise ValueError("duplicate task_id")
        task_ids.add(task.task_id)
        for point in task.point_ids:
            key = (task.field_id, point)
            if key not in required:
                raise ValueError("unknown field or point")
            if key in seen:
                raise ValueError("point assigned to multiple regions")
            seen.add(key)
    if seen != required:
        raise ValueError("partition leaves uncovered task points")


def normalize_visit_weights(weights: Sequence[float]) -> tuple[float, ...]:
    """P->D per-field visit weights; no implicit clipping or zero-size visits.

These are target area shares, not guaranteed realized grid areas. The region
builder must separately report realized areas and enforce exact ownership.
"""
    if len(weights) == 0:
        raise ValueError("at least one visit is required")
    values = tuple(float(w) for w in weights)
    if any(not isfinite(w) or w <= 0 for w in values):
        raise ValueError("visit weights must be finite and strictly positive")
    scale = max(values)
    scaled = tuple(w / scale for w in values)
    if any(w == 0 for w in scaled):
        raise ValueError("visit weights exceed representable dynamic range")
    total = sum(scaled)
    return tuple(w / total for w in scaled)
