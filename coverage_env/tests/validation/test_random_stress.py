"""Deterministic randomized transition stress test over multiple fields."""

import random

from coverage_env.tests.validation.support import (
    ValidationTestCase, capture_step, finite_path, make_action, make_environment,
    path_length,
)


class RandomStressTests(ValidationTestCase):
    def test_seeded_random_action_sequences_preserve_invariants(self):
        seeds = (7, 19, 2025)
        last_transition = None
        for seed in seeds:
            randomizer = random.Random(seed)
            environment = make_environment(field_count=4, max_energy=1_000_000.0)
            for step_index in range(12):
                eligible = [
                    field for field in environment.fields
                    if field.get_uncovered_points()
                ]
                if not eligible:
                    break
                field = randomizer.choice(eligible)
                action = make_action(
                    field.field_id,
                    entry=f"B{randomizer.randrange(4)}",
                    exit=f"B{randomizer.randrange(4)}",
                    ratio=randomizer.uniform(0.05, 0.22),
                )
                before, result, after = capture_step(environment, action)
                last_transition = (before, action, result, after)
                uav_state = after["uavs"]["UAV-1"]
                self.assert_transition(
                    result["coverage"] > 0.0
                    and result["distance"] >= 0.0
                    and result["energy"] >= 0.0
                    and uav_state["remaining_energy"] >= 0.0
                    and uav_state["total_distance"] + 1e-8
                        >= before["uavs"]["UAV-1"]["total_distance"]
                    and finite_path(result["path"])
                    and abs(path_length(result["path"]) - result["distance"]) < 1e-6,
                    before=before, action=action, result=result, after=after,
                    message=f"Random stress invariant failed (seed={seed}, step={step_index}).",
                )
        self.assertIsNotNone(last_transition)
