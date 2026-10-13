"""Depot location and spatial distance helpers."""

from __future__ import annotations

from math import hypot, isfinite
from typing import Tuple, Union

from shapely.geometry import Point

Position2D = Tuple[float, float]


class Depot:
    """A UAV depot represented by a local metric ``(x, y)`` coordinate."""

    def __init__(self, depot_id: Union[str, int], position: Position2D) -> None:
        """Create a depot and validate its local XY position."""
        self.depot_id = depot_id
        self.position = self._validate_position(position)

    @staticmethod
    def _validate_position(position: Position2D) -> Position2D:
        """Convert a two-dimensional coordinate to finite float values."""
        if len(position) != 2:
            raise ValueError("position must contain exactly (x, y).")
        x, y = float(position[0]), float(position[1])
        if not isfinite(x) or not isfinite(y):
            raise ValueError("position coordinates must be finite.")
        return x, y

    def get_position(self) -> Position2D:
        """Return the depot's local XY coordinate."""
        return self.position

    def distance_to(self, point: Position2D) -> float:
        """Return the Euclidean distance to a local XY point, in meters."""
        x, y = self._validate_position(point)
        return hypot(x - self.position[0], y - self.position[1])

    def distance_to_field(self, field) -> float:
        """Return the shortest distance from the depot to a field polygon.

        The field's ``xy_polygon`` must use the same local metric coordinate
        frame as this depot. The result is zero when the depot lies inside it.
        """
        if not hasattr(field, "xy_polygon"):
            raise TypeError("field must provide an xy_polygon in local XY coordinates.")
        return float(Point(self.position).distance(field.xy_polygon))

    def __repr__(self) -> str:
        return f"Depot(depot_id={self.depot_id!r}, position={self.position!r})"
