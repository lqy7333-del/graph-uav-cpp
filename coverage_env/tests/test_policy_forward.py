"""Forward tests for workload-free spatial action decoding."""

from __future__ import annotations

import unittest

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.action_space import CoverageAction
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.decoder.action_spatial import ACTION_SPATIAL_DIM


class CoveragePolicyForwardTests(unittest.TestCase):
    """Verify action triplets, masking, sampling, and environment execution."""

    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG_PATH)
        self.environment = InstanceGenerator(self.config).generate(
            num_fields=5, seed=20261012
        )
        self.environment.reset()
        self.actions = self.environment.get_action_candidates()
        self.policy = CoveragePolicy(self.environment, config=self.config)
        self.policy.eval()

    def test_state_to_action_distribution_and_execution(self):
        state = self.environment.get_state()
        output = self.policy(state)
        expected_count = sum(
            len(field.boundaries) ** 2
            for field in self.environment.fields if field.get_uncovered_points()
        )
        self.assertEqual(len(output["actions"]), expected_count)
        self.assertTrue(all(isinstance(a, CoverageAction) for a in output["actions"]))
        self.assertTrue(all(not hasattr(a, "workload_ratio") for a in output["actions"]))
        self.assertEqual(output["scores"].shape, (expected_count, 1))
        # Derived rather than hardcoded: action embedding + per-action spatial
        # distances + the shared global context.
        expected_input_dim = (
            self.policy.action_embedding.output_dim
            + ACTION_SPATIAL_DIM
            + self.policy.encoder.hidden_dim
        )
        self.assertEqual(
            output["action_input"].shape, (expected_count, expected_input_dim)
        )
        self.assertEqual(output["encoder_output"]["field_embedding"].shape[1], 128)
        self.assertEqual(output["encoder_output"]["boundary_embedding"].shape[1], 128)
        self.assertTrue(bool(output["mask"].any()))
        self.assertTrue(torch.all(output["masked_scores"][~output["mask"]] == -torch.inf))
        self.assertTrue(torch.all(output["action_distribution"][~output["mask"]] == 0.0))
        self.assertAlmostEqual(float(output["action_distribution"].sum()), 1.0, places=6)

        selected = self.policy.sample_action(output)
        self.assertIsInstance(selected, CoverageAction)
        selected_index = output["actions"].index(selected)
        self.assertTrue(bool(output["mask"][selected_index]))
        next_state, reward, done, info = self.environment.step(selected)
        self.assertIsInstance(next_state, dict)
        self.assertIsInstance(reward, float)
        self.assertGreater(info["estimated_workload"], 0.0)
        self.assertGreater(info["coverage_gain"], 0.0)
        self.assertEqual(next_state["uavs"]["UAV-1"]["position"], info["exit_point"])

        print(
            f"Spatial actions={expected_count}, feasible={int(output['mask'].sum())}, "
            f"probability_sum={float(output['action_distribution'].sum()):.6f}"
        )
        self.policy.print_top_k(output, k=3)
        rows = self.policy.top_k_actions(output, k=3)
        self.assertTrue(rows)
        self.assertIn("probability", rows[0])
        self.assertNotIn("workload", rows[0])

    def test_energy_mask_sets_invalid_probabilities_to_zero(self):
        self.environment.uavs[0].remaining_energy = 0.0
        output = self.policy(self.environment.get_state())
        self.assertFalse(bool(output["mask"].any()))
        self.assertTrue(torch.all(output["action_distribution"] == 0.0))


if __name__ == "__main__":
    unittest.main()
