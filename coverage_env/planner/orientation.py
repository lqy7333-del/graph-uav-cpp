"""Finite-candidate scan orientation selection for coverage execution."""

from __future__ import annotations

from math import atan2, cos, hypot, pi, radians, sin
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coverage_env.env.boundary import Boundary
    from coverage_env.env.field import Field

Point2D = tuple[float, float]


def _normalize_axis_angle(angle: float) -> float:
    """Normalize an undirected scan axis to degrees in ``[0, 180)``."""
    return angle % 180.0


def _axis_angle_difference(first: float, second: float) -> float:
    """Return the smallest angular difference between two undirected axes."""
    difference = abs(_normalize_axis_angle(first) - _normalize_axis_angle(second))
    return min(difference, 180.0 - difference)


def _normal_angle(boundary: "Boundary") -> float:
    """Return the boundary normal's undirected angle in degrees."""
    return _normalize_axis_angle(
        atan2(boundary.normal_vector[1], boundary.normal_vector[0]) * 180.0 / pi
    )


def select_coverage_orientation(
    field: "Field",
    entry_boundary: "Boundary",
    exit_boundary: "Boundary",
    entry_point: Point2D,
) -> float:
    """Choose among the field axis, its perpendicular, and exit normal.

    Candidate scoring combines deviation from the field's long axis, the
    angle and scan-line offset at the entry, and the angle and scan-line offset
    at the exit boundary. For a same-boundary entry and exit, the boundary
    normal is selected directly as requested.
    """
    if entry_boundary.id == exit_boundary.id:
        return _normal_angle(entry_boundary)

    field_axis = _normalize_axis_angle(field.orientation)
    candidates = [
        field_axis,
        _normalize_axis_angle(field_axis + 90.0),
        _normal_angle(exit_boundary),
    ]
    distinct_candidates: list[float] = []
    for angle in candidates:
        if not any(_axis_angle_difference(angle, other) < 1e-9
                   for other in distinct_candidates):
            distinct_candidates.append(angle)

    min_x, min_y, max_x, max_y = field.xy_polygon.bounds
    scale = max(hypot(max_x - min_x, max_y - min_y), 1.0)
    entry_normal = (-entry_boundary.normal_vector[0], -entry_boundary.normal_vector[1])
    exit_normal = exit_boundary.normal_vector
    exit_point = exit_boundary.center
    samples = [point.coordinate for point in field.grid_points]
    if not samples:
        samples = [(float(x), float(y)) for x, y in field.xy_polygon.exterior.coords[:-1]]

    scored = []
    for angle in distinct_candidates:
        theta = radians(angle)
        axis = (cos(theta), sin(theta))
        normal = (-axis[1], axis[0])
        entry_angular_cost = 1.0 - abs(axis[0] * entry_normal[0] + axis[1] * entry_normal[1])
        exit_angular_cost = 1.0 - abs(axis[0] * exit_normal[0] + axis[1] * exit_normal[1])
        entry_offset = min(
            abs((x - entry_point[0]) * normal[0] + (y - entry_point[1]) * normal[1])
            for x, y in samples
        )
        exit_offset = min(
            abs((x - exit_point[0]) * normal[0] + (y - exit_point[1]) * normal[1])
            for x, y in samples
        )
        score = (
            0.45 * _axis_angle_difference(angle, field_axis) / 90.0 * scale
            + 0.20 * entry_angular_cost * scale
            + 0.20 * exit_angular_cost * scale
            + 0.10 * entry_offset
            + 0.10 * exit_offset
        )
        scored.append((score, angle))
    return min(scored)[1]
