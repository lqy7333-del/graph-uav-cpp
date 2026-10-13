"""Validation for the RL-ready state/action/reward environment contract."""

from __future__ import annotations

import unittest

from coverage_env.config import energy_parameters
from coverage_env.env.depot import Depot
from coverage_env.env.environment import CoverageEnvironment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.geometry.discretization import discretize_field
from coverage_env.planner.configuration import CoverageConfiguration
from coverage_env.rl.state_encoder import StateEncoder


def build_environment():
    """Build a single rectangular field with a sufficiently charged UAV."""
    transformer = CoordinateTransformer(31.23, 121.47)
    coords = [transformer.to_latlon(x, y) for x, y in
              [(0, 0), (40, 0), (40, 30), (0, 30)]]
    field = Field("field-1", coords)
    discretize_field(field, 5.0)
    uav = UAV("UAV-1", 100_000, 8, 5, 2, energy_consumption_parameters=energy_parameters())
    env = CoverageEnvironment(
        [field], Depot("depot-1", (-10.0, -10.0)), [uav],
        rl_config={
            "workload_options": [0.1, 0.2, 0.3, 0.5],
            "reward_weights": {"lambda1": 0.001, "lambda2": 0.01, "lambda3": 0.1},
        },
    )
    return env


class RLInterfaceTests(unittest.TestCase):
    """Check observations, candidate expansion, masks, and transitions."""

    def test_reset_candidates_mask_and_step(self):
        env = build_environment()
        state = env.reset()
        self.assertIn("uavs", state)
        self.assertIn("fields", state)
        self.assertEqual(len(state["boundaries"]), 4)
        self.assertIn("center", state["fields"]["field-1"])

        actions = env.get_action_candidates()
        self.assertEqual(len(actions), 4 * 4)
        mask = env.get_action_mask(actions)
        self.assertEqual(len(mask), len(actions))
        self.assertTrue(any(mask))
        from coverage_env.rl.action_mask import get_action_mask
        from coverage_env.rl.action_space import ActionGenerator
        self.assertEqual(len(ActionGenerator().generate(state)), len(actions))
        self.assertEqual(get_action_mask(actions, state), mask)

        selected = CoverageConfiguration("field-1", "B0", "B2", 0.1)
        next_state, reward, done, info = env.step(selected)
        self.assertIsInstance(next_state, dict)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)
        self.assertEqual(info["selected_field"], "field-1")
        self.assertGreater(info["coverage_gain"], 0.0)
        self.assertEqual(next_state["uavs"]["UAV-1"]["position"], info["exit_point"])

        encoded = StateEncoder.encode(next_state)
        self.assertEqual(len(encoded["field_features"]), 1)
        self.assertEqual(len(encoded["boundary_features"]), 4)

    def test_mask_filters_energy_infeasible_actions_without_mutation(self):
        env = build_environment()
        env.reset()
        actions = env.get_action_candidates()
        before_covered = env.fields[0].calculate_coverage_area()
        env.uavs[0].remaining_energy = 0.01
        mask = env.get_action_mask(actions)
        self.assertFalse(any(mask))
        self.assertEqual(env.fields[0].calculate_coverage_area(), before_covered)


if __name__ == "__main__":
    unittest.main()
