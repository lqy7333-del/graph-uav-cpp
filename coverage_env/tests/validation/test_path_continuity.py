"""Validate route endpoints, per-step continuity, and distance feedback."""

from coverage_env.tests.validation.support import (
    ValidationTestCase, capture_step, finite_path, make_action, make_environment,
    path_length,
)


class PathContinuityTests(ValidationTestCase):
    def test_route_starts_at_uav_ends_at_exit_and_matches_distance(self):
        environment = make_environment()
        action = make_action()
        before, result, after = capture_step(environment, action)
        path = result["path"]
        self.assert_transition(
            len(path) >= 2
            and path[0] == before["uavs"]["UAV-1"]["position"]
            and path[-1] == result["exit_point"]
            and finite_path(path)
            and abs(path_length(path) - result["distance"]) < 1e-6
            and after["uavs"]["UAV-1"]["position"] == path[-1],
            before=before, action=action, result=result, after=after,
            message="Executed path is discontinuous or feedback length is wrong.",
        )
