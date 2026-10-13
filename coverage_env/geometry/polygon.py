"""Validation helpers for WGS84 polygons."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

from shapely.geometry import Polygon

LatLon = Tuple[float, float]
PolygonInput = Union[Polygon, Sequence[LatLon]]


def wgs84_polygon_from_coordinates(polygon: PolygonInput) -> Polygon:
    """Build a valid WGS84 Shapely polygon from latitude/longitude pairs.

    Plain coordinate sequences use ``(latitude, longitude)`` order. A Shapely
    polygon follows its standard ``(longitude, latitude)`` coordinate order.
    Existing polygon holes are preserved. Invalid/self-intersecting geometry
    is rejected rather than silently repaired.
    """
    if isinstance(polygon, Polygon):
        result = polygon
    else:
        points = list(polygon)
        if len(points) < 3:
            raise ValueError("A field polygon needs at least three coordinate pairs.")
        lon_lat = []
        for index, point in enumerate(points):
            if len(point) != 2:
                raise ValueError(
                    f"Polygon coordinate {index} must be (latitude, longitude)."
                )
            latitude, longitude = float(point[0]), float(point[1])
            if not -90.0 <= latitude <= 90.0:
                raise ValueError(f"Latitude must be in [-90, 90], got {latitude}.")
            if not -180.0 <= longitude <= 180.0:
                raise ValueError(f"Longitude must be in [-180, 180], got {longitude}.")
            lon_lat.append((longitude, latitude))
        result = Polygon(lon_lat)

    if result.is_empty or not result.is_valid or result.area <= 0:
        raise ValueError("The WGS84 field polygon must be valid and have positive area.")
    return result
