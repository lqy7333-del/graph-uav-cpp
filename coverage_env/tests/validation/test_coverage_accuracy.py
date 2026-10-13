"""Validate workload coverage against remaining discretized field area."""

from coverage_env.tests.validation.support import (
    ValidationTestCase, capture_step, make_action, make_environment,
)


class CoverageAccuracyTests(ValidationTestCase):
    def test_workload_area_is_met_with_at_most_one_cell_quantization(self):
        environment = make_environment()
        field = environment.get_field("field-1")
        available_area = sum(point.point_area for point in field.get_uncovered_points())
        action = make_action(ratio=0.3)
        before, result, after = capture_step(environment, action)
        requested_area = 0.3 * available_area
        largest_cell = max(point.point_area for point in field.grid_points)
        self.assert_transition(
            requested_area - 1e-8 <= result["coverage"]
            <= requested_area + largest_cell + 1e-8
            and abs(
                before["fields"]["field-1"]["remaining_area"]
                - after["fields"]["field-1"]["remaining_area"]
                - result["coverage"]
            ) < 1e-8,
            before=before, action=action, result=result, after=after,
            message="Workload coverage differs from requested remaining-area fraction.",
        )
