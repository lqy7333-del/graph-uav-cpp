"""WGS84 latitude/longitude to local metric coordinate conversions."""

from __future__ import annotations

from math import floor, isfinite
from typing import Iterable, Tuple

from pyproj import CRS, Transformer

LatLon = Tuple[float, float]
Point2D = Tuple[float, float]


class CoordinateTransformer:
    """Convert ``(latitude, longitude)`` to local ``(x, y)`` meters and back.

    A UTM CRS is selected from the origin for ordinary latitudes. A local
    azimuthal-equidistant CRS is used in polar regions outside UTM's range.
    The projected origin is subtracted so the local coordinate frame is near
    ``(0, 0)``. The public argument order is always latitude then longitude,
    even though PROJ's ``always_xy`` transform internally uses longitude first.
    """

    def __init__(self, origin_latitude: float, origin_longitude: float) -> None:
        """Create a local projection centered at a WGS84 origin."""
        self._validate_latlon(origin_latitude, origin_longitude)
        self.origin_latitude = float(origin_latitude)
        self.origin_longitude = float(origin_longitude)
        self.geographic_crs = CRS.from_epsg(4326)
        self.projected_crs = self._select_projected_crs(
            self.origin_latitude, self.origin_longitude
        )
        self._forward = Transformer.from_crs(
            self.geographic_crs, self.projected_crs, always_xy=True
        )
        self._inverse = Transformer.from_crs(
            self.projected_crs, self.geographic_crs, always_xy=True
        )
        self._origin_x, self._origin_y = self._forward.transform(
            self.origin_longitude, self.origin_latitude
        )

    @staticmethod
    def _validate_latlon(latitude: float, longitude: float) -> None:
        """Reject invalid WGS84 coordinates before passing them to PROJ."""
        if not isfinite(latitude) or not isfinite(longitude):
            raise ValueError("Latitude and longitude must be finite numbers.")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError(f"Latitude must be in [-90, 90], got {latitude}.")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError(f"Longitude must be in [-180, 180], got {longitude}.")

    @staticmethod
    def _select_projected_crs(latitude: float, longitude: float) -> CRS:
        """Choose the local UTM zone, falling back to AEQD near the poles."""
        if -80.0 <= latitude <= 84.0:
            zone = min(60, max(1, floor((longitude + 180.0) / 6.0) + 1))
            epsg = (32600 if latitude >= 0.0 else 32700) + zone
            return CRS.from_epsg(epsg)
        return CRS.from_proj4(
            f"+proj=aeqd +lat_0={latitude} +lon_0={longitude} "
            "+datum=WGS84 +units=m +no_defs"
        )

    def to_xy(self, latitude: float, longitude: float) -> Point2D:
        """Project one WGS84 coordinate to local ``(x, y)`` meters."""
        self._validate_latlon(latitude, longitude)
        x, y = self._forward.transform(longitude, latitude)
        return float(x - self._origin_x), float(y - self._origin_y)

    def to_latlon(self, x: float, y: float) -> LatLon:
        """Convert local ``(x, y)`` meters to ``(latitude, longitude)``."""
        if not isfinite(x) or not isfinite(y):
            raise ValueError("Local x and y must be finite numbers.")
        longitude, latitude = self._inverse.transform(
            x + self._origin_x, y + self._origin_y
        )
        return float(latitude), float(longitude)

    def to_xy_many(self, coordinates: Iterable[LatLon]) -> list[Point2D]:
        """Project an iterable of ``(latitude, longitude)`` pairs."""
        return [self.to_xy(latitude, longitude) for latitude, longitude in coordinates]

    def to_latlon_many(self, coordinates: Iterable[Point2D]) -> list[LatLon]:
        """Inverse-project an iterable of local ``(x, y)`` pairs."""
        return [self.to_latlon(x, y) for x, y in coordinates]


def wgs84_to_xy(
    latitude: float, longitude: float, transformer: CoordinateTransformer
) -> Point2D:
    """Convenience wrapper for converting with an existing local frame."""
    return transformer.to_xy(latitude, longitude)


def xy_to_wgs84(x: float, y: float, transformer: CoordinateTransformer) -> LatLon:
    """Convenience wrapper for inverse conversion with an existing local frame."""
    return transformer.to_latlon(x, y)
