"""Shared fixtures and failure diagnostics for validation tests."""

from __future__ import annotations

import json
from math import hypot, isfinite
import unittest

from coverage_env.config import energy_parameters
from coverage_env.env.depot import Depot
from coverage_env.env.environment import CoverageEnvironment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.geometry.discretization import discretize_field
from coverage_env.planner.configuration import CoverageConfiguration


def make_environment(
    *, field_count: int = 1, max_energy: float = 100_000.0, resolution: float = 5.0
) -> CoverageEnvironment:
    """Build separated rectangular fields and discretize them for a test."""
    geographic_frame = CoordinateTransformer(31.23, 121.47)
    fields = []
    for index in range(field_count):
        offset = index * 100.0
        coordinates = [
            geographic_frame.to_latlon(offset, 0.0),
            geographic_frame.to_latlon(offset + 40.0, 0.0),
            geographic_frame.to_latlon(offset + 40.0, 30.0),
            geographic_frame.to_latlon(offset, 30.0),
        ]
        fields.append(Field(f"field-{index + 1}", coordinates))
    depot = Depot("depot-1", (-10.0, -20.0))
    uav = UAV(
        "UAV-1", max_energy=max_energy, speed=8.0, coverage_width=5.0,
        sensor_width=2.0,
        energy_consumption_parameters=energy_parameters(),
    )
    environment = CoverageEnvironment(fields, depot, [uav])
    for field in environment.fields:
        discretize_field(field, resolution)
    environment.reset()
    return environment


def make_action(field_id="field-1", entry="B0", exit="B2", ratio=0.3):
    """Create a conventional rectangular-field action."""
    return CoverageConfiguration(field_id, entry, exit, ratio)


def diagnostic(before, action, result, after) -> str:
    """Format the required transition context for failed assertions."""
    return "\n".join((
        "Transition diagnostic:",
        f"before state: {_format(before)}",
        f"action: {_format(action)}",
        f"result: {_format(result)}",
        f"after state: {_format(after)}",
    ))


def _format(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=repr)


class ValidationTestCase(unittest.TestCase):
    """Assertions that always attach a full transition diagnostic."""

    def assert_transition(self, condition, *, before, action, result, after, message):
        self.assertTrue(
            condition,
            f"{message}\n{diagnostic(before, action, result, after)}",
        )


def capture_step(environment, action):
    """Execute an action and turn unexpected exceptions into diagnostics."""
    before = environment.get_state()
    try:
        result = environment.step(action)
    except Exception as exc:
        after = environment.get_state()
        raise AssertionError(diagnostic(
            before, action, f"{type(exc).__name__}: {exc}", after
        )) from exc
    return before, result, environment.get_state()


def capture_uav_path(environment, path, *, area=0.0):
    """Execute a manual UAV path and preserve the same failure diagnostics."""
    before = environment.get_state()
    try:
        result = environment.uavs[0].execute_path(path, area=area)
    except Exception as exc:
        after = environment.get_state()
        raise AssertionError(diagnostic(
            before, path, f"{type(exc).__name__}: {exc}", after
        )) from exc
    return before, result, environment.get_state()


def path_length(path) -> float:
    """Calculate polyline length for feedback consistency checks."""
    return sum(
        hypot(end[0] - start[0], end[1] - start[1])
        for start, end in zip(path[:-1], path[1:])
    )


def finite_path(path) -> bool:
    """Check that all waypoints are finite XY coordinate pairs."""
    return all(len(point) == 2 and all(isfinite(value) for value in point) for point in path)
