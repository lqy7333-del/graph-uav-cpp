"""Tests for WGS84 and local XY coordinate conversions."""

import unittest

from coverage_env.geometry.coordinate import CoordinateTransformer


class CoordinateTransformerTests(unittest.TestCase):
    """Round-trip checks for coordinate projection and validation."""

    def test_round_trip(self):
        transformer = CoordinateTransformer(31.2304, 121.4737)
        xy = transformer.to_xy(31.2310, 121.4750)
        actual = transformer.to_latlon(*xy)
        self.assertAlmostEqual(actual[0], 31.2310, places=7)
        self.assertAlmostEqual(actual[1], 121.4750, places=7)
        self.assertLess(abs(xy[0]), 500)
        self.assertLess(abs(xy[1]), 500)

    def test_origin_maps_to_local_zero(self):
        transformer = CoordinateTransformer(-33.8, 151.2)
        self.assertAlmostEqual(transformer.to_xy(-33.8, 151.2)[0], 0.0, places=7)
        self.assertAlmostEqual(transformer.to_xy(-33.8, 151.2)[1], 0.0, places=7)

    def test_invalid_latitude_is_rejected(self):
        transformer = CoordinateTransformer(0, 0)
        with self.assertRaises(ValueError):
            transformer.to_xy(91, 0)


if __name__ == "__main__":
    unittest.main()
