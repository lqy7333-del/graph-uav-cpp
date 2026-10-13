"""Uniform polygon grid discretization for later coverage-region planning."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from math import ceil, isfinite
from typing import TYPE_CHECKING, Tuple

from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from coverage_env.env.field import Field

Point2D = Tuple[float, float]


@dataclass
class GridPoint:
    """A discrete coverage sample in local metric coordinates.

    ``point_area`` is the area represented by the point. A regular grid uses
    ``resolution ** 2`` for every retained point; consequently the summed area
    is a resolution-dependent estimate near polygon boundaries.
    """

    id: str
    x: float
    y: float
    field_id: str | int
    point_area: float
    covered: bool = False
    neighbors: list[str] = dataclass_field(default_factory=list)
    row: int = 0
    column: int = 0

    @property
    def coordinate(self) -> Point2D:
        """Return the local XY coordinate as a pair."""
        return self.x, self.y

    @property
    def area(self) -> float:
        """Alias for ``point_area`` for generic spatial-area consumers."""
        return self.point_area


@dataclass(frozen=True)
class GridCell:
    """One clipped grid cell that intersects the field polygon."""

    row: int
    column: int
    geometry: BaseGeometry
    center: Point2D
    area: float


def discretize_polygon(
    polygon: Polygon, resolution: float, *, max_cells: int = 1_000_000
) -> list[GridCell]:
    """Clip a square grid to ``polygon`` and return non-empty cells in meters.

    ``resolution`` is the side length of each grid square. Returned geometry
    is clipped to the field, so edge cells can have less area than a full cell.
    """
    if resolution <= 0:
        raise ValueError("resolution must be greater than zero.")
    min_x, min_y, max_x, max_y = polygon.bounds
    columns = ceil((max_x - min_x) / resolution)
    rows = ceil((max_y - min_y) / resolution)
    if rows * columns > max_cells:
        raise ValueError(
            f"Requested grid has {rows * columns} candidate cells; "
            f"limit is {max_cells}. Increase resolution."
        )

    cells: list[GridCell] = []
    for row in range(rows):
        y0 = min_y + row * resolution
        y1 = min(y0 + resolution, max_y)
        for column in range(columns):
            x0 = min_x + column * resolution
            x1 = min(x0 + resolution, max_x)
            clipped = polygon.intersection(box(x0, y0, x1, y1))
            if clipped.is_empty or clipped.area <= 0.0:
                continue
            center = clipped.centroid
            cells.append(
                GridCell(
                    row=row,
                    column=column,
                    geometry=clipped,
                    center=(float(center.x), float(center.y)),
                    area=float(clipped.area),
                )
            )
    return cells


def discretize_field(
    field: "Field", resolution: float, *, max_points: int = 2_000_000
) -> list[GridPoint]:
    """Sample a field with a regular grid and attach points to the ``Field``.

    Grid samples are placed at the center of each resolution-sized cell. Only
    points strictly inside ``field.xy_polygon`` are retained. Each point
    represents ``resolution ** 2`` square meters. Eight-neighborhood links
    are populated after filtering, so neighbors always refer to retained IDs.
    """
    if not isfinite(resolution) or resolution <= 0.0:
        raise ValueError("resolution must be a finite number greater than zero.")
    if max_points <= 0:
        raise ValueError("max_points must be greater than zero.")

    min_x, min_y, max_x, max_y = field.xy_polygon.bounds
    columns = ceil((max_x - min_x) / resolution)
    rows = ceil((max_y - min_y) / resolution)
    candidate_count = rows * columns
    if candidate_count > max_points:
        raise ValueError(
            f"Grid requires {candidate_count} candidate points, exceeding "
            f"max_points={max_points}; increase resolution."
        )

    points: list[GridPoint] = []
    by_index: dict[tuple[int, int], GridPoint] = {}
    represented_area = resolution * resolution
    for row in range(rows):
        y = min_y + (row + 0.5) * resolution
        for column in range(columns):
            x = min_x + (column + 0.5) * resolution
            if not field.xy_polygon.contains(Point(x, y)):
                continue
            point = GridPoint(
                id=f"{field.field_id}:r{row}:c{column}",
                x=float(x),
                y=float(y),
                field_id=field.field_id,
                point_area=represented_area,
                row=row,
                column=column,
            )
            points.append(point)
            by_index[(row, column)] = point

    for (row, column), point in by_index.items():
        for row_offset in (-1, 0, 1):
            for column_offset in (-1, 0, 1):
                if row_offset == 0 and column_offset == 0:
                    continue
                neighbor = by_index.get((row + row_offset, column + column_offset))
                if neighbor is not None:
                    point.neighbors.append(neighbor.id)

    field.grid_points = points
    field.grid_resolution = float(resolution)
    return points
