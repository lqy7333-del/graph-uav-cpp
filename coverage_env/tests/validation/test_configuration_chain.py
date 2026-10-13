"""Validate chained ``state_t, action, state_(t+1)`` transitions."""

from coverage_env.tests.validation.support import (
    ValidationTestCase, capture_step, make_action, make_environment,
)


class ConfigurationChainTests(ValidationTestCase):
    def test_three_configurations_chain_without_reset(self):
        environment = make_environment(field_count=3)
        actions = [
            make_action("field-1", "B0", "B2", 0.25),
            make_action("field-2", "B1", "B3", 0.30),
            make_action("field-3", "B0", "B0", 0.35),
        ]
        previous_after = environment.get_state()
        for action in actions:
            before, result, after = capture_step(environment, action)
            self.assert_transition(
                before == previous_after
                and result["state"] == after
                and after["uavs"]["UAV-1"]["total_distance"]
                    >= before["uavs"]["UAV-1"]["total_distance"]
                and after["uavs"]["UAV-1"]["remaining_energy"]
                    <= before["uavs"]["UAV-1"]["remaining_energy"],
                before=before, action=action, result=result, after=after,
                message="Configuration sequence did not preserve chained state.",
            )
            previous_after = after
