"""Validate that only the selected field's coverage state changes."""

from coverage_env.tests.validation.support import (
    ValidationTestCase, capture_step, make_action, make_environment,
)


class CoverageUpdateTests(ValidationTestCase):
    def test_action_marks_selected_field_points_covered(self):
        environment = make_environment(field_count=2)
        action = make_action("field-1", "B0", "B2", 0.3)
        before, result, after = capture_step(environment, action)
        first_after = after["fields"]["field-1"]
        second_after = after["fields"]["field-2"]
        self.assert_transition(
            first_after["covered_area"] > before["fields"]["field-1"]["covered_area"]
            and first_after["uncovered_points"] < before["fields"]["field-1"]["uncovered_points"]
            and second_after == before["fields"]["field-2"]
            and abs(first_after["covered_area"] - result["coverage"]) < 1e-9,
            before=before, action=action, result=result, after=after,
            message="Coverage update was missing or affected the wrong field.",
        )
