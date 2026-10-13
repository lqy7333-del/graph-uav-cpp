"""Regression tests for separating boundary entry and coverage anchor."""

import unittest
from shapely.geometry import LineString, Point

from coverage_env.config import energy_parameters
from coverage_env.env.depot import Depot
from coverage_env.env.environment import Environment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.geometry.discretization import discretize_field
from coverage_env.planner.configuration import CoverageConfiguration


class EntryAnchorDifferenceTests(unittest.TestCase):
    """Ensure each action has an explicit boundary-to-sweep transit leg."""

    def setUp(self):
        transformer = CoordinateTransformer(31.23, 121.47)
        vertices = [(0, 0), (60, 0), (60, 40), (0, 40)]
        polygon = [transformer.to_latlon(x, y) for x, y in vertices]
        self.field = Field("field-1", polygon)
        discretize_field(self.field, 5.0)
        entry = self.field.get_boundary("B0")
        self.depot = Depot(
            "depot",
            (entry.center[0] + entry.normal_vector[0] * 12.0,
             entry.center[1] + entry.normal_vector[1] * 12.0),
        )
        self.uav = UAV(
            "uav-1", max_energy=100_000, speed=8, coverage_width=5,
            sensor_width=2,
            energy_consumption_parameters=energy_parameters(),
        )
        self.environment = Environment([self.field], self.depot, [self.uav])
        self.environment.reset()

    def test_boundary_entry_is_optimized_separately_from_coverage_anchor(self):
        entry_boundary = self.field.get_boundary("B0")
        edge_line = LineString([
            entry_boundary.start_point, entry_boundary.end_point
        ])
        # Simulate a later mission step: the edge-adjacent cells have already
        # been covered, so the next sweep anchor must lie in the field interior.
        for point in self.field.grid_points:
            if edge_line.distance(Point(point.x, point.y)) <= 7.5:
                point.covered = True
        action = CoverageConfiguration("field-1", "B0", "B3", 0.3)
        feedback = self.environment.step_configuration(action)
        result = self.environment.execution_records[-1]
        boundary_entry = result["entry_boundary_point"]
        anchor = result["coverage_start_point"]
        entry_line = LineString([
            self.field.get_boundary("B0").start_point,
            self.field.get_boundary("B0").end_point,
        ])

        self.assertLess(entry_line.distance(Point(boundary_entry)), 1e-8)
        self.assertEqual(result["coverage_path"][0], anchor)
        self.assertEqual(result["entry_transit_path"][0], self.depot.position)
        self.assertEqual(result["entry_transit_path"][1], boundary_entry)
        self.assertEqual(result["entry_transit_path"][-1], anchor)
        self.assertNotEqual(boundary_entry, anchor)
        self.assertGreaterEqual(len(result["entry_transit_path"]), 3)
        self.assertEqual(feedback["path"], result["total_path"])

    def test_next_entry_is_nearest_to_previous_exit(self):
        first = CoverageConfiguration("field-1", "B0", "B3", 0.3)
        second = CoverageConfiguration("field-1", "B1", "B2", 0.3)
        self.environment.step_configuration(first)
        previous_exit = self.environment.execution_records[-1]["exit_boundary_point"]
        self.environment.step_configuration(second)
        actual_entry = self.environment.execution_records[-1]["entry_boundary_point"]
        boundary = self.field.get_boundary("B1")
        segment = LineString([boundary.start_point, boundary.end_point])
        self.assertLess(segment.distance(Point(actual_entry)), 1e-8)
        second_result = self.environment.execution_records[-1]
        self.assertLess(
            LineString([boundary.start_point, boundary.end_point]).distance(
                Point(actual_entry)
            ),
            1e-8,
        )
        self.assertEqual(second_result["coverage_path"][0], second_result["coverage_start_point"])
        self.assertEqual(second_result["entry_transit_path"][0], previous_exit)
        self.assertEqual(second_result["entry_transit_path"][-1], second_result["coverage_start_point"])


if __name__ == "__main__":
    unittest.main()
