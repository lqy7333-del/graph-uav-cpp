"""Tests for configuration instantiation, transition feedback, and plotting."""

import tempfile
import unittest
from math import hypot
from pathlib import Path

import matplotlib
from shapely.geometry import LineString, Point

matplotlib.use("Agg")

from coverage_env.config import energy_parameters
from coverage_env.env.depot import Depot
from coverage_env.env.environment import Environment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.geometry.discretization import discretize_field
from coverage_env.planner.configuration import CoverageConfiguration
from coverage_env.planner.configuration_executor import ConfigurationExecutor
from coverage_env.planner.coverage_path import (
    generate_boustrophedon_path,
    plan_boustrophedon_path,
)
from coverage_env.visualization.plot import plot_configuration_execution


def field_from_local_vertices(field_id, vertices):
    """Build a WGS84 field from metric vertices near Shanghai."""
    transformer = CoordinateTransformer(31.23, 121.47)
    coordinates = [transformer.to_latlon(x, y) for x, y in vertices]
    return Field(field_id, coordinates)


class ConfigurationExecutorTests(unittest.TestCase):
    """Exercise a high-level action against a small discretized field."""

    def setUp(self):
        self.field = field_from_local_vertices(
            "field-1", [(0, 0), (40, 0), (40, 30), (0, 30)]
        )
        discretize_field(self.field, resolution=5.0)
        entry_boundary = self.field.get_boundary("B0")
        self.depot = Depot(
            "depot-1",
            (
                entry_boundary.center[0] + entry_boundary.normal_vector[0] * 10.0,
                entry_boundary.center[1] + entry_boundary.normal_vector[1] * 10.0,
            ),
        )
        self.uav = UAV(
            "UAV-1",
            max_energy=100_000.0,
            speed=8.0,
            coverage_width=5.0,
            sensor_width=2.0,
            energy_consumption_parameters=energy_parameters(),
        )
        self.environment = Environment([self.field], self.depot, [self.uav])
        self.environment.reset()
        self.configuration = CoverageConfiguration("field-1", "B0", "B3", 0.3)

    def test_executor_generates_region_path_and_energy_feedback(self):
        executor = ConfigurationExecutor(grid_resolution=5.0)
        result = executor.execute(self.configuration, self.environment)

        self.assertTrue(result.region)
        self.assertGreater(len(result.coverage_path), 2)
        self.assertEqual(result.coverage_path[0], result.coverage_start_point)
        self.assertEqual(result.entry_transit_path[0], self.depot.position)
        self.assertEqual(result.entry_transit_path[1], result.entry_boundary_point)
        self.assertEqual(result.entry_transit_path[-1], result.coverage_start_point)
        self.assertEqual(result.exit_transit_path[0], result.coverage_path[-1])
        self.assertEqual(result.exit_transit_path[-1], result.exit_boundary_point)
        self.assertEqual(result.total_path[0], self.depot.position)
        self.assertEqual(result.total_path[-1], result.exit_boundary_point)
        self.assertAlmostEqual(
            result.distance,
            sum(
                ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
                for start, end in zip(result.total_path[:-1], result.total_path[1:])
            ),
        )
        self.assertAlmostEqual(result.covered_area,
                               sum(point.point_area for point in result.region))
        self.assertGreaterEqual(
            result.covered_area,
            0.3 * sum(point.point_area for point in self.field.grid_points),
        )
        region_ids = {point.id for point in result.region}
        point_by_id = {point.id: point for point in result.region}
        reached = set()
        pending = [result.region[0].id]
        while pending:
            point_id = pending.pop()
            if point_id in reached:
                continue
            reached.add(point_id)
            pending.extend(
                neighbor_id
                for neighbor_id in point_by_id[point_id].neighbors
                if neighbor_id in region_ids and neighbor_id not in reached
            )
        self.assertEqual(reached, region_ids)
        self.assertGreater(result.distance, 0.0)
        self.assertGreater(result.energy, 0.0)
        self.assertAlmostEqual(
            result.energy,
            self.uav.alpha * result.distance + self.uav.beta * result.covered_area,
        )
        exit_boundary = self.field.get_boundary("B3")
        entry_boundary = self.field.get_boundary("B0")
        entry_line = LineString([entry_boundary.start_point, entry_boundary.end_point])
        self.assertLessEqual(entry_line.distance(Point(result.entry_point)), 1e-8)
        self.assertGreaterEqual(
            hypot(
                result.coverage_start_point[0] - result.entry_boundary_point[0],
                result.coverage_start_point[1] - result.entry_boundary_point[1],
            ),
            0.0,
        )
        self.assertLessEqual(
            LineString([exit_boundary.start_point, exit_boundary.end_point]).distance(
                Point(result.exit_point)
            ),
            1e-8,
        )
        self.assertTrue(all(point.covered for point in result.region))
        print(
            "configuration: entry=%s, points=%d, path=%.2f m, exit=%s, "
            "area=%.2f m^2, energy=%.2f Wh"
            % (result.entry_point, len(result.region), result.distance,
               result.exit_point, result.covered_area, result.energy)
        )

    def test_workload_fraction_uses_remaining_area(self):
        initial_area = sum(point.point_area for point in self.field.grid_points)
        first = self.environment.step_configuration(self.configuration)
        remaining_after_first = initial_area - first["coverage"]
        second = self.environment.step_configuration(self.configuration)

        # Workload is rounded up to complete scan bands for geometric continuity.
        self.assertAlmostEqual(first["coverage"], 0.3 * initial_area, delta=50.0)
        self.assertAlmostEqual(second["coverage"], 0.3 * remaining_after_first, delta=50.0)
        self.assertGreater(second["coverage"], 0.0)

    def test_rectangular_region_uses_parallel_alternating_lanes(self):
        plan = plan_boustrophedon_path(
            self.field.grid_points,
            orientation=0.0,
            start_point=self.field.get_boundary("B0").center,
            lane_spacing=5.0,
            field_polygon=self.field.xy_polygon,
            entry_boundary=self.field.get_boundary("B0"),
        )
        self.assertGreaterEqual(len(plan.strokes), 6)
        self.assertTrue(all(abs(start[1] - end[1]) < 0.1 for start, end in plan.strokes))
        self.assertTrue(all(start != end for start, end in plan.strokes))
        self.assertTrue(all(self.field.xy_polygon.buffer(1e-6).covers(
            LineString(connector)
        ) for connector in plan.connectors))

    def test_environment_step_feedback_and_reset(self):
        feedback = self.environment.step_configuration(self.configuration)
        self.assertGreater(feedback["distance"], 0.0)
        self.assertGreater(feedback["energy"], 0.0)
        self.assertGreater(feedback["coverage"], 0.0)
        self.assertEqual(feedback["new_state"]["uavs"]["UAV-1"]["position"],
                         feedback["exit_point"])
        self.assertGreater(feedback["new_state"]["fields"]["field-1"]["covered_area"], 0.0)

        self.environment.reset()
        self.assertTrue(all(not point.covered for point in self.field.grid_points))
        self.assertEqual(self.uav.current_position, self.depot.position)

    def test_execution_plot_is_saved(self):
        result = ConfigurationExecutor().execute(self.configuration, self.environment)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "configuration.png"
            figure, _ = plot_configuration_execution(
                self.field,
                result.entry_point,
                result.exit_point,
                result.region,
                result.coverage_path,
                entry_transit_path=result.entry_transit_path,
                exit_transit_path=result.exit_transit_path,
                coverage_start_point=result.coverage_start_point,
                coverage_strokes=result.coverage_strokes,
                scan_transit_paths=result.scan_transit_paths,
                growth_edge_id=result.growth_edge_id,
                remaining_points=result.remaining_points,
                show=False,
                save_path=str(path),
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            matplotlib.pyplot.close(figure)


if __name__ == "__main__":
    unittest.main()
