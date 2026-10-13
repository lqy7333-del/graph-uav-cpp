"""Run and visualize one simulated coverage-configuration execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from coverage_env.config import energy_parameters
from coverage_env.env.depot import Depot
from coverage_env.env.environment import Environment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.planner.configuration import CoverageConfiguration
from coverage_env.visualization.plot import plot_configuration_execution


def build_demo_field() -> Field:
    """Return an irregular sample parcel around Shanghai, in WGS84."""
    return Field(
        field_id="demo-field-1",
        polygon=[
            (31.23000, 121.47000),
            (31.23010, 121.47220),
            (31.23075, 121.47255),
            (31.23125, 121.47160),
            (31.23100, 121.47015),
            (31.23000, 121.47000),
        ],
    )


def main(*, show: bool = False) -> Path:
    """Execute a sample configuration and save its visualization.

    Pass ``show=True`` to open the figure in an interactive Matplotlib window.
    The PNG is always written next to this module so the result remains
    available after tests or the process finish.
    """
    field = build_demo_field()
    entry_boundary = field.get_boundary("B0")
    depot_offset = 20.0
    depot_position = (
        entry_boundary.center[0] + entry_boundary.normal_vector[0] * depot_offset,
        entry_boundary.center[1] + entry_boundary.normal_vector[1] * depot_offset,
    )
    depot = Depot("depot-1", depot_position)
    uav = UAV(
        "UAV-1",
        max_energy=100_000.0,
        speed=8.0,
        coverage_width=5.0,
        sensor_width=2.0,
        energy_consumption_parameters=energy_parameters(),
    )
    environment = Environment([field], depot, [uav])
    environment.reset()

    configuration = CoverageConfiguration(
        field_id=field.field_id,
        entry_boundary_id="B0",
        exit_boundary_id="B3",
        workload_ratio=0.3,
    )
    feedback = environment.step_configuration(configuration)
    output_path = Path(__file__).with_name("configuration_execution.png")
    plot_configuration_execution(
        field,
        feedback["entry_point"],
        feedback["exit_point"],
        [point for point in field.grid_points if point.covered],
        feedback["coverage_path"],
        entry_transit_path=feedback["entry_transit_path"],
        exit_transit_path=feedback["exit_transit_path"],
        coverage_strokes=environment.execution_records[-1]["coverage_strokes"],
        scan_transit_paths=feedback["scan_transit_paths"],
        growth_edge_id=feedback["growth_edge_id"],
        remaining_points=feedback["remaining_points"],
        entry_boundary_id=configuration.entry_boundary_id,
        exit_boundary_id=configuration.exit_boundary_id,
        show=show,
        save_path=str(output_path),
    )
    state = feedback["new_state"]["uavs"][uav.uav_id]
    print(f"Field: {field.field_id} ({field.area:.2f} m^2)")
    print(f"Entry point: {feedback['entry_point']}")
    print(f"Exit point: {feedback['exit_point']}")
    print(f"Path length: {feedback['distance']:.2f} m")
    print(f"Covered area: {feedback['coverage']:.2f} m^2")
    print(f"Energy consumption: {feedback['energy']:.2f} Wh")
    print(f"Remaining energy: {state['remaining_energy']:.2f} Wh")
    print(f"Visualization saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show",
        action="store_true",
        help="also open the visualization in an interactive window",
    )
    arguments = parser.parse_args()
    main(show=arguments.show)
