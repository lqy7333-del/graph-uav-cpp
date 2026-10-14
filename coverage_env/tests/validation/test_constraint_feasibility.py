"""Validate infeasible actions fail without partially changing the state."""

from coverage_env.planner.configuration import CoverageConfiguration
from coverage_env.tests.validation.support import (
    ValidationTestCase, make_action, make_environment,
)


class ConstraintFeasibilityTests(ValidationTestCase):
    def test_invalid_boundary_and_insufficient_energy_return_failure_transition(self):
        environment = make_environment(max_energy=100_000.0)
        invalid_action = make_action(entry="B99")
        before = environment.get_state()
        next_state, reward, done, info = environment.step(invalid_action)
        after = environment.get_state()
        self.assert_transition(
            info["failed"] and reward < 0.0 and before == after == next_state
            and not done,
            before=before, action=invalid_action, result=info, after=after,
            message="Invalid boundary should return a non-mutating failure transition.",
        )

        low_energy = make_environment(max_energy=0.01)
        action = make_action()
        before = low_energy.get_state()
        next_state, reward, done, info = low_energy.step(action)
        after = low_energy.get_state()
        self.assert_transition(
            info["failed"] and reward < 0.0 and before == after == next_state
            and not done,
            before=before, action=action, result=info, after=after,
            message="Insufficient energy should return an atomic failure transition.",
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
