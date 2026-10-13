"""Geographic field polygon and its derived local geometry."""

from __future__ import annotations

from math import atan2, degrees, hypot, radians, cos, sin
from typing import Sequence, Tuple, Union

from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from coverage_env.env.boundary import Boundary
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.geometry.polygon import wgs84_polygon_from_coordinates

Point2D = Tuple[float, float]
LatLon = Tuple[float, float]
PolygonInput = Union[Polygon, Sequence[LatLon]]


class Field:
    """A WGS84 farm parcel with a local metric polygon and edge boundaries.

    Coordinate sequences are supplied as ``(latitude, longitude)`` pairs.
    A Shapely ``Polygon`` is also accepted; its coordinates follow the usual
    ``(longitude, latitude)`` X/Y convention. Local XY coordinates are meters
    in a UTM projection selected from the parcel's approximate center.
    """

    def __init__(self, field_id: Union[str, int], polygon: PolygonInput) -> None:
        """Validate and derive all geometry for one field."""
        self.field_id = field_id
        self.polygon_wgs84 = wgs84_polygon_from_coordinates(polygon)
        representative = list(self.polygon_wgs84.exterior.coords)[:-1]
        origin_lon = sum(point[0] for point in representative) / len(representative)
        origin_lat = sum(point[1] for point in representative) / len(representative)
        self.coordinate_transformer = CoordinateTransformer(origin_lat, origin_lon)

        self.grid_points = []
        self.grid_resolution = None
        self._derive_local_geometry()

    def set_coordinate_transformer(
        self, transformer: CoordinateTransformer
    ) -> None:
        """Reproject this field into a shared environment XY coordinate frame.

        Existing discretization is rebuilt at its previous resolution because
        point coordinates and boundaries belong to the old frame.
        """
        if not isinstance(transformer, CoordinateTransformer):
            raise TypeError("transformer must be a CoordinateTransformer.")
        resolution = self.grid_resolution
        self.coordinate_transformer = transformer
        self._derive_local_geometry()
        self.grid_points = []
        self.grid_resolution = None
        if resolution is not None:
            from coverage_env.geometry.discretization import discretize_field

            discretize_field(self, resolution)

    def _derive_local_geometry(self) -> None:
        """Refresh local polygon metrics and boundaries from WGS84 geometry."""
        exterior = self._project_ring(self.polygon_wgs84.exterior.coords)
        interiors = [
            self._project_ring(interior.coords)
            for interior in self.polygon_wgs84.interiors
        ]
        # CCW exterior and CW holes make the right-hand normal point outward.
        self.xy_polygon = orient(Polygon(exterior, interiors), sign=1.0)
        if not self.xy_polygon.is_valid or self.xy_polygon.area <= 0:
            raise ValueError("The projected field polygon is invalid or has zero area.")

        self.area = float(self.xy_polygon.area)
        self.perimeter = float(self.xy_polygon.length)
        centroid = self.xy_polygon.centroid
        self.centroid: Point2D = (float(centroid.x), float(centroid.y))
        self.bounding_box: Tuple[float, float, float, float] = tuple(
            float(value) for value in self.xy_polygon.bounds
        )  # min_x, min_y, max_x, max_y
        self.bbox = self.bounding_box
        self.orientation = self._calculate_orientation()
        angle = radians(self.orientation)
        self.orientation_vector: Point2D = (cos(angle), sin(angle))
        self.boundaries = self._make_boundaries()

    def _project_ring(self, coordinates) -> list[Point2D]:
        """Project a Shapely ring (longitude, latitude) into local XY."""
        return [
            self.coordinate_transformer.to_xy(latitude=lat, longitude=lon)
            for lon, lat, *_ in coordinates
        ]

    def _calculate_orientation(self) -> float:
        """Return the long-axis angle from the minimum rotated rectangle.

        The result is degrees counter-clockwise from east, normalized to
        ``[-90, 90)`` because a field axis has no direction.
        """
        rectangle = self.xy_polygon.minimum_rotated_rectangle
        corners = list(rectangle.exterior.coords)
        edges = [
            (corners[i + 1][0] - corners[i][0],
             corners[i + 1][1] - corners[i][1])
            for i in range(len(corners) - 1)
        ]
        dx, dy = max(edges, key=lambda edge: hypot(*edge))
        angle = degrees(atan2(dy, dx))
        return (angle + 90.0) % 180.0 - 90.0

    def _make_boundaries(self) -> list[Boundary]:
        """Split exterior and interior rings into individually addressable edges."""
        rings = [(self.xy_polygon.exterior, False)] + [
            (ring, True) for ring in self.xy_polygon.interiors
        ]
        boundaries: list[Boundary] = []
        for ring_id, (ring, is_hole) in enumerate(rings):
            points = list(ring.coords)
            for start, end in zip(points[:-1], points[1:]):
                boundaries.append(
                    Boundary.from_endpoints(
                        boundary_id=f"B{len(boundaries)}",
                        start_point=(float(start[0]), float(start[1])),
                        end_point=(float(end[0]), float(end[1])),
                        ring_id=ring_id,
                        is_hole=is_hole,
                    )
                )
        return boundaries

    def get_boundary(self, boundary_id: str) -> Boundary:
        """Return a boundary by its stable id, raising ``KeyError`` if absent."""
        for boundary in self.boundaries:
            if boundary.id == boundary_id:
                return boundary
        raise KeyError(f"Field {self.field_id!r} has no boundary {boundary_id!r}.")

    def summary(self) -> dict:
        """Return the field's principal geometry values in a serializable mapping."""
        return {
            "field_id": self.field_id,
            "area_m2": self.area,
            "perimeter_m": self.perimeter,
            "centroid_xy_m": self.centroid,
            "bounding_box_xy_m": self.bounding_box,
            "orientation_deg": self.orientation,
            "boundary_count": len(self.boundaries),
        }

    def get_points_inside(self, region: BaseGeometry) -> list:
        """Return grid points contained by a local-XY Shapely geometry."""
        if not isinstance(region, BaseGeometry):
            raise TypeError("region must be a Shapely geometry in local XY coordinates.")
        return [
            point for point in self.grid_points
            if region.covers(Point(point.x, point.y))
        ]

    def get_uncovered_points(self) -> list:
        """Return the grid points whose ``covered`` state is false."""
        return [point for point in self.grid_points if not point.covered]

    def calculate_coverage_area(self) -> float:
        """Estimate covered area by summing areas represented by covered points."""
        return float(sum(point.point_area for point in self.grid_points if point.covered))

    def __repr__(self) -> str:
        return (
            f"Field(field_id={self.field_id!r}, area={self.area:.2f} m^2, "
            f"boundaries={len(self.boundaries)})"
        )
