"""Tests for field geometry, boundary extraction, and grid discretization."""

import unittest

from coverage_env.env.field import Field
from coverage_env.env.environment import FieldEnvironment
from coverage_env.geometry.discretization import discretize_polygon


class FieldTests(unittest.TestCase):
    """Geometry behavior checks using a small simulated Shanghai farm."""

    def setUp(self):
        self.field = Field(
            "test-field",
            [
                (31.2300, 121.4700),
                (31.2300, 121.4720),
                (31.2310, 121.4720),
                (31.2310, 121.4700),
            ],
        )

    def test_metric_properties_and_boundaries(self):
        self.assertGreater(self.field.area, 0)
        self.assertGreater(self.field.perimeter, 0)
        self.assertEqual(len(self.field.boundaries), 4)
        self.assertAlmostEqual(self.field.boundaries[0].length,
                               self.field.boundaries[2].length, delta=1.0)
        self.assertAlmostEqual(self.field.orientation, 0.0, delta=2.0)
        self.assertEqual(self.field.get_boundary("B0").id, "B0")
        self.assertAlmostEqual(self.field.summary()["area_m2"], self.field.area)

    def test_outward_normals_and_grid(self):
        for boundary in self.field.boundaries:
            self.assertAlmostEqual(
                boundary.tangent_vector[0] * boundary.normal_vector[0]
                + boundary.tangent_vector[1] * boundary.normal_vector[1],
                0.0,
                places=7,
            )
        cells = discretize_polygon(self.field.xy_polygon, resolution=20.0)
        self.assertTrue(cells)
        self.assertAlmostEqual(sum(cell.area for cell in cells), self.field.area,
                               delta=self.field.area * 1e-8)

    def test_invalid_polygon_is_rejected(self):
        with self.assertRaises(ValueError):
            Field("bad", [(31.0, 121.0), (31.0, 121.0), (31.0, 121.0)])

    def test_environment_field_registry(self):
        environment = FieldEnvironment([self.field])
        self.assertIs(environment.get_field("test-field"), self.field)
        self.assertEqual(environment.fields, (self.field,))
        with self.assertRaises(ValueError):
            environment.add_field(self.field)


if __name__ == "__main__":
    unittest.main()
