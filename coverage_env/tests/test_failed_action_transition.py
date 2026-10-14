"""Failure transitions remain atomic and do not interrupt PPO collection."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from coverage_env.rl.action_space import CoverageAction
from coverage_env.tests.test_rl_interface import build_environment


class FailedActionTransitionTests(unittest.TestCase):
    """Check failed selected actions return state-preserving transitions."""

    def test_failed_execution_rolls_back_partial_uav_mutation(self):
        environment = build_environment()
        environment.reset()
        action = CoverageAction("field-1", "B0", "B2")
        before = environment.get_state()

        def mutate_then_fail(configuration):
            environment.uavs[0].execute_path(
                [environment.uavs[0].current_position, (10.0, 0.0)], area=0.0
            )
            raise ValueError("synthetic exit-path resource failure")

        with (
            patch.object(environment, "estimate_workload", return_value=0.5),
            patch.object(environment, "_refine_workload_for_execution", return_value=0.5),
            patch.object(environment, "step_configuration", side_effect=mutate_then_fail),
        ):
            next_state, reward, done, info = environment.step(action)

        self.assertEqual(next_state, before)
        self.assertEqual(environment.get_state(), before)
        self.assertLess(reward, 0.0)
        self.assertFalse(done)
        self.assertTrue(info["failed"])
        self.assertEqual(info["energy"], 0.0)
        self.assertEqual(info["payload"], 0.0)
        self.assertEqual(environment.execution_records, ())
        self.assertEqual(len(environment.episode_logger.transitions), 1)

    def test_failure_threshold_ends_only_the_episode(self):
        environment = build_environment()
        environment.rl_config["failure_termination_threshold"] = 2
        environment.reset()
        action = CoverageAction("field-1", "B99", "B2")

        first = environment.step(action)
        second = environment.step(action)

        self.assertFalse(first[2])
        self.assertTrue(second[2])
        self.assertTrue(first[3]["failed"] and second[3]["failed"])
        self.assertEqual(first[0], second[0])
        self.assertEqual(len(environment.episode_logger.transitions), 2)


if __name__ == "__main__":
    unittest.main()
