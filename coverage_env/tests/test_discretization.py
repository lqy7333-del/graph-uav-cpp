"""Tests for regular field grid points, adjacency, and spatial queries."""

import unittest

from shapely.geometry import Point, box

from coverage_env.env.field import Field
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.geometry.discretization import GridPoint, discretize_field


def field_from_local_vertices(field_id, vertices):
    """Build a WGS84 field from vertices defined in meters around Shanghai."""
    transformer = CoordinateTransformer(31.23, 121.47)
    coordinates = [transformer.to_latlon(x, y) for x, y in vertices]
    return Field(field_id, coordinates)


class DiscretizationTests(unittest.TestCase):
    """Exercise rectangular and irregular fields at several resolutions."""

    def setUp(self):
        self.fields = [
            field_from_local_vertices(
                "rectangle",
                [(0, 0), (40, 0), (40, 20), (0, 20)],
            ),
            field_from_local_vertices(
                "irregular",
                [(0, 0), (40, 0), (40, 20), (25, 32), (0, 25)],
            ),
        ]

    def test_resolution_counts_area_error_and_polygon_membership(self):
        for field in self.fields:
            for resolution in (2.0, 5.0, 10.0):
                points = discretize_field(field, resolution)
                estimated_area = sum(point.point_area for point in points)
                absolute_error = abs(estimated_area - field.area)
                relative_error = absolute_error / field.area * 100.0
                print(
                    f"{field.field_id}: resolution={resolution:g} m, "
                    f"count={len(points)}, estimated={estimated_area:.2f} m^2, "
                    f"true={field.area:.2f} m^2, error={relative_error:.2f}%"
                )

                self.assertTrue(points)
                self.assertIs(field.grid_points, points)
                self.assertEqual(field.grid_resolution, resolution)
                self.assertTrue(
                    all(point.point_area == resolution * resolution for point in points)
                )
                self.assertTrue(
                    all(field.xy_polygon.contains(Point(point.x, point.y)) for point in points)
                )

    def test_gridpoint_adjacency_and_coverage_queries(self):
        field = self.fields[0]
        points = discretize_field(field, resolution=5.0)
        self.assertIsInstance(points[0], GridPoint)
        self.assertEqual(points[0].field_id, field.field_id)
        self.assertEqual(points[0].coordinate, (points[0].x, points[0].y))
        self.assertEqual(points[0].area, points[0].point_area)

        point_by_id = {point.id: point for point in points}
        for point in points:
            self.assertLessEqual(len(point.neighbors), 8)
            self.assertNotIn(point.id, point.neighbors)
            for neighbor_id in point.neighbors:
                self.assertIn(neighbor_id, point_by_id)
                self.assertIn(point.id, point_by_id[neighbor_id].neighbors)

        region = box(points[0].x - 0.1, points[0].y - 0.1,
                     points[0].x + 0.1, points[0].y + 0.1)
        inside = field.get_points_inside(region)
        self.assertEqual([point.id for point in inside], [points[0].id])
        points[0].covered = True
        self.assertEqual(field.get_uncovered_points(), points[1:])
        self.assertAlmostEqual(field.calculate_coverage_area(), 25.0)

    def test_invalid_resolution_is_rejected(self):
        with self.assertRaises(ValueError):
            discretize_field(self.fields[0], 0)


if __name__ == "__main__":
    unittest.main()
