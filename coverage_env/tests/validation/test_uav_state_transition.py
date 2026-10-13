"""Validate a UAV's position, distance, energy, and area state transition."""

from coverage_env.tests.validation.support import (
    ValidationTestCase, capture_uav_path, make_environment, path_length,
)


class UAVStateTransitionTests(ValidationTestCase):
    def test_manual_path_updates_dynamic_state(self):
        environment = make_environment()
        uav = environment.uavs[0]
        action = [(uav.current_position), (uav.current_position[0] + 30.0,
                                            uav.current_position[1] + 40.0)]
        before, result, after = capture_uav_path(environment, action, area=125.0)
        distance = path_length(action)
        expected_energy = uav.alpha * distance + uav.beta * 125.0
        self.assert_transition(
            uav.current_position == action[-1]
            and abs(uav.total_distance - distance) < 1e-9
            and abs(uav.remaining_energy - (uav.max_energy - expected_energy)) < 1e-9
            and abs(uav.covered_area - 125.0) < 1e-9,
            before=before, action=action, result=result, after=after,
            message="UAV did not apply the expected path state transition.",
        )
