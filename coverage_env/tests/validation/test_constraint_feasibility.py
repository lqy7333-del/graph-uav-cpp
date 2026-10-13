"""Validate infeasible actions fail without partially changing the state."""

from coverage_env.planner.configuration import CoverageConfiguration
from coverage_env.tests.validation.support import (
    ValidationTestCase, make_action, make_environment,
)


class ConstraintFeasibilityTests(ValidationTestCase):
    def test_invalid_boundary_and_insufficient_energy_are_atomic(self):
        environment = make_environment(max_energy=100_000.0)
        invalid_action = make_action(entry="B99")
        before = environment.get_state()
        error = None
        try:
            environment.step(invalid_action)
        except (KeyError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        after = environment.get_state()
        self.assert_transition(
            error is not None and before == after,
            before=before, action=invalid_action, result=error, after=after,
            message="Invalid boundary should fail before modifying state.",
        )

        low_energy = make_environment(max_energy=0.01)
        action = make_action()
        before = low_energy.get_state()
        error = None
        try:
            low_energy.step(action)
        except ValueError as exc:
            error = f"{type(exc).__name__}: {exc}"
        after = low_energy.get_state()
        self.assert_transition(
            error is not None and before == after,
            before=before, action=action, result=error, after=after,
            message="Insufficient energy should fail atomically.",
        )

    def test_configuration_rejects_out_of_range_workload(self):
        environment = make_environment()
        before = environment.get_state()
        action = {"field_id": "field-1", "workload_ratio": 1.2}
        error = None
        try:
            CoverageConfiguration("field-1", "B0", "B2", 1.2)
        except ValueError as exc:
            error = f"{type(exc).__name__}: {exc}"
        after = environment.get_state()
        self.assert_transition(
            error is not None and before == after,
            before=before, action=action, result=error, after=after,
            message="Out-of-range workload should be rejected without state change.",
        )
