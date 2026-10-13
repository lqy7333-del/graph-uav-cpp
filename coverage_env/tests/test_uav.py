"""Tests for depot distance, UAV execution, and environment reset behavior."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from coverage_env.config import energy_parameters
from coverage_env.env.depot import Depot
from coverage_env.env.environment import Environment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.visualization.plot import plot_uav


class UAVSimulationTests(unittest.TestCase):
    """Verify the first-stage distance and linear energy model."""

    def make_uav(self, *, max_energy=500.0):
        """Create a UAV using the config-driven energy coefficients."""
        return UAV(
            uav_id="UAV-1",
            max_energy=max_energy,
            speed=10.0,
            coverage_width=5.0,
            sensor_width=2.0,
            energy_consumption_parameters=energy_parameters(),
        )

    def test_depot_and_manual_path_execution(self):
        depot = Depot("depot-1", (0.0, 0.0))
        uav = self.make_uav()
        environment = Environment(fields=[], depot=depot, uavs=[uav])

        initial = environment.reset()[uav.uav_id]
        self.assertEqual(initial["position"], depot.get_position())
        self.assertEqual(uav.current_position, (0.0, 0.0))
        self.assertAlmostEqual(depot.distance_to((3, 4)), 5.0)

        path = [(0, 0), (100, 0), (100, 100)]
        result = uav.execute_path(path)
        self.assertAlmostEqual(result["path_length"], 200.0)
        self.assertAlmostEqual(result["energy_consumption"], 40.0)
        self.assertAlmostEqual(result["remaining_energy"], 460.0)
        self.assertEqual(result["final_position"], (100.0, 100.0))
        self.assertAlmostEqual(uav.total_distance, 200.0)
        self.assertEqual(uav.current_path, [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0)])
        self.assertEqual(uav.get_state()["position"], (100.0, 100.0))
        print(
            "manual path: length=%.2f m, energy=%.2f Wh, remaining=%.2f Wh, "
            "final=%s"
            % (result["path_length"], result["energy_consumption"],
               result["remaining_energy"], result["final_position"])
        )

    def test_coverage_energy_reset_and_environment_state(self):
        depot = Depot("depot-1", (10.0, 20.0))
        uav = self.make_uav()
        environment = Environment(fields=[], depot=depot, uavs=[uav])
        environment.reset()
        result = uav.execute_path([(10, 20), (20, 20)], area=1000.0)
        self.assertAlmostEqual(
            result["energy_consumption"], uav.alpha * 10.0 + uav.beta * 1000.0
        )
        self.assertAlmostEqual(uav.covered_area, 1000.0)
        uav.visited_fields.append("field-1")

        state = environment.reset()[uav.uav_id]
        self.assertEqual(state["position"], (10.0, 20.0))
        self.assertEqual(state["remaining_energy"], uav.max_energy)
        self.assertEqual(state["total_distance"], 0.0)
        self.assertEqual(state["covered_area"], 0.0)
        self.assertEqual(uav.visited_fields, [])
        self.assertEqual(uav.current_path, [])

    def test_depot_distance_to_field(self):
        field = Field(
            "field-1",
            [(31.0, 121.0), (31.0, 121.001), (31.001, 121.001), (31.001, 121.0)],
        )
        depot = Depot("inside", field.centroid)
        self.assertAlmostEqual(depot.distance_to_field(field), 0.0)

    def test_insufficient_energy_does_not_partially_execute(self):
        uav = self.make_uav(max_energy=5.0)
        uav.reset((0.0, 0.0))
        before = uav.get_state().copy()
        with self.assertRaises(ValueError):
            uav.execute_path([(0, 0), (100, 0)])
        self.assertEqual(uav.get_state(), before)
        self.assertEqual(uav.current_path, [])

    def test_plot_uav_saves_image(self):
        depot = Depot("depot-1", (0.0, 0.0))
        uav = self.make_uav()
        uav.reset(depot.position)
        uav.execute_path([(0, 0), (100, 0), (100, 100)])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "uav.png"
            figure, _ = plot_uav(depot, uav, show=False, save_path=str(path))
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            matplotlib.pyplot.close(figure)


if __name__ == "__main__":
    unittest.main()
