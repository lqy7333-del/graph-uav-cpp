"""Validate per-step and cumulative energy accounting."""

from coverage_env.tests.validation.support import (
    ValidationTestCase, capture_step, make_action, make_environment,
)


class EnergyConsistencyTests(ValidationTestCase):
    def test_energy_matches_distance_and_covered_area(self):
        environment = make_environment()
        uav = environment.uavs[0]
        action = make_action()
        before, result, after = capture_step(environment, action)
        expected = uav.alpha * result["distance"] + uav.beta * result["coverage"]
        energy_delta = (
            before["uavs"][uav.uav_id]["remaining_energy"]
            - after["uavs"][uav.uav_id]["remaining_energy"]
        )
        cumulative_energy = uav.total_energy_consumed
        self.assert_transition(
            abs(result["energy"] - expected) < 1e-8
            and abs(energy_delta - result["energy"]) < 1e-8
            and abs(cumulative_energy - result["energy"]) < 1e-8,
            before=before, action=action, result=result, after=after,
            message="Energy is inconsistent with distance, area, or battery delta.",
        )
