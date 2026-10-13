"""End-to-end three-field mission simulation and visualization test."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from coverage_env.config import energy_parameters
from coverage_env.env.depot import Depot
from coverage_env.env.environment import CoverageEnvironment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.planner.configuration import CoverageConfiguration
from coverage_env.visualization.plot import plot_mission


def build_three_field_mission() -> tuple[CoverageEnvironment, list[CoverageConfiguration]]:
    """Build separated rectangular parcels in one geographic area."""
    geographic_frame = CoordinateTransformer(31.23, 121.47)
    fields = []
    for index, offset_x in enumerate((0.0, 100.0, 200.0), start=1):
        vertices = [
            geographic_frame.to_latlon(offset_x, 0.0),
            geographic_frame.to_latlon(offset_x + 40.0, 0.0),
            geographic_frame.to_latlon(offset_x + 40.0, 30.0),
            geographic_frame.to_latlon(offset_x, 30.0),
        ]
        fields.append(Field(f"field-{index}", vertices))

    # Depot shares the first field's local XY frame, as required by Environment.
    depot = Depot("depot-1", (-10.0, -20.0))
    uav = UAV(
        "UAV-1",
        max_energy=100_000.0,
        speed=8.0,
        coverage_width=5.0,
        sensor_width=2.0,
        energy_consumption_parameters=energy_parameters(),
    )
    environment = CoverageEnvironment(fields, depot, [uav])
    configurations = [
        CoverageConfiguration("field-1", "B0", "B2", 0.4),
        CoverageConfiguration("field-2", "B1", "B3", 0.5),
        CoverageConfiguration("field-3", "B0", "B0", 0.8),
    ]
    return environment, configurations


def run_mission(*, show: bool = False, save_path: str | None = None) -> dict:
    """Run the three requested actions and optionally save the mission plot."""
    environment, configurations = build_three_field_mission()
    environment.reset()
    for configuration in configurations:
        feedback = environment.step(configuration)
        print(
            f"{configuration.field_id}: coverage={feedback['coverage']:.1f} m^2, "
            f"distance={feedback['distance']:.1f} m, "
            f"energy={feedback['energy']:.1f} Wh"
        )
    summary = environment.get_summary()
    if save_path is not None or show:
        plot_mission(
            environment,
            show=show,
            save_path=save_path,
        )
    print(
        "Mission summary: "
        f"distance={summary['total_distance']:.1f} m, "
        f"energy={summary['total_energy']:.1f} Wh, "
        f"coverage={summary['coverage_ratio']:.1%}, "
        f"completed_fields={summary['completed_fields']}, "
        f"remaining_energy={summary['remaining_energy']}, "
        f"trajectory_points={len(summary['uav_trajectory'].get('UAV-1', []))}"
    )
    return summary


class MissionTests(unittest.TestCase):
    """Verify reset, sequential actions, summary, and full mission plot."""

    def test_three_configuration_mission_and_plot(self):
        environment, configurations = build_three_field_mission()
        initial_state = environment.reset()
        self.assertEqual(len(initial_state["fields"]), 3)
        self.assertEqual(initial_state["uavs"]["UAV-1"]["position"],
                         environment.depot.position)

        feedbacks = [environment.step(configuration) for configuration in configurations]
        self.assertEqual(len(environment.execution_records), 3)
        self.assertTrue(all(feedback["coverage"] > 0.0 for feedback in feedbacks))
        self.assertEqual(
            environment.get_state()["uavs"]["UAV-1"]["position"],
            feedbacks[-1]["exit_point"],
        )

        summary = environment.get_summary()
        self.assertEqual(summary["execution_count"], 3)
        self.assertGreater(summary["total_distance"], 0.0)
        self.assertGreater(summary["total_energy"], 0.0)
        self.assertGreater(summary["coverage_ratio"], 0.0)
        self.assertLessEqual(summary["coverage_ratio"], 1.0)
        expected_trajectory_length = sum(
            len(feedback["path"]) for feedback in feedbacks
        ) - (len(feedbacks) - 1)
        self.assertEqual(
            len(summary["uav_trajectory"]["UAV-1"]), expected_trajectory_length
        )

        # Reprojection puts the separated fields in one shared coordinate frame.
        centers = [field.centroid[0] for field in environment.fields]
        self.assertEqual(centers, sorted(centers))
        self.assertGreater(centers[1] - centers[0], 80.0)
        self.assertGreater(centers[2] - centers[1], 80.0)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            plot_path = Path(temp_dir) / "mission.png"
            figure, _ = plot_mission(
                environment, show=False, save_path=str(plot_path)
            )
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 1000)
            matplotlib.pyplot.close(figure)

        print(
            "Mission summary: "
            f"distance={summary['total_distance']:.1f} m, "
            f"energy={summary['total_energy']:.1f} Wh, "
            f"coverage={summary['coverage_ratio']:.1%}"
        )


if __name__ == "__main__":
    output = Path(__file__).parents[1] / "mission_execution.png"
    run_mission(save_path=str(output))
    print(f"Mission visualization saved to: {output}")
    unittest.main(argv=[__file__])
