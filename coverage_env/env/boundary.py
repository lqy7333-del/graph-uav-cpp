"""Field boundary segments used as high-level coverage entry/exit choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class Boundary:
    """A straight segment on a field ring, represented in local metric XY.

    ``normal_vector`` points out of the field. For holes, it points from the
    covered field area into the hole. Boundary identifiers are stable within
    the owning :class:`Field` and are formatted as ``B0``, ``B1``, and so on.
    """

    id: str
    start_point: Point2D
    end_point: Point2D
    length: float
    center: Point2D
    tangent_vector: Point2D
    normal_vector: Point2D
    ring_id: int = 0
    is_hole: bool = False

    @classmethod
    def from_endpoints(
        cls,
        boundary_id: str,
        start_point: Point2D,
        end_point: Point2D,
        *,
        ring_id: int = 0,
        is_hole: bool = False,
    ) -> "Boundary":
        """Build a segment and its unit tangent and outward normal vectors."""
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length <= 0.0:
            raise ValueError("A boundary segment must have non-zero length.")

        tangent = (dx / length, dy / length)
        # Field rings are oriented with the field area on their left side.
        normal = (tangent[1], -tangent[0])
        center = ((start_point[0] + end_point[0]) / 2.0,
                  (start_point[1] + end_point[1]) / 2.0)
        return cls(
            id=boundary_id,
            start_point=start_point,
            end_point=end_point,
            length=length,
            center=center,
            tangent_vector=tangent,
            normal_vector=normal,
            ring_id=ring_id,
            is_hole=is_hole,
        )
