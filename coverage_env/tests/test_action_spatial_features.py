"""Per-action spatial features must make the nearest-field decision decidable."""

from __future__ import annotations

import unittest
from math import hypot

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.decoder.action_spatial import (
    ACTION_SPATIAL_DIM,
    build_action_spatial_feature,
)


class ActionSpatialFeatureTests(unittest.TestCase):
    """The actor previously had no way to tell which candidate field was closest."""

    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG_PATH)
        self.config["environment"]["num_fields"] = {"min": 6, "max": 6}
        self.environment = InstanceGenerator(self.config).generate(num_fields=6, seed=7)
        self.state = self.environment.reset()
        torch.manual_seed(0)
        self.policy = CoveragePolicy(self.environment, config=self.config)
        self.policy.eval()
        self.output = self.policy(self.state, environment=self.environment)

    def test_shape_range_and_environment_state_equivalence(self):
        actions = self.output["actions"]
        from_env = build_action_spatial_feature(actions, self.environment)
        from_state = build_action_spatial_feature(actions, self.state)
        self.assertEqual(tuple(from_env.shape), (len(actions), ACTION_SPATIAL_DIM))
        self.assertTrue(torch.allclose(from_env, from_state))
        self.assertGreaterEqual(float(from_env.min()), 0.0)
        self.assertLessEqual(float(from_env.max()), 1.0)

    def test_uav_to_entry_column_ranks_candidates_by_true_distance(self):
        """The whole point: the feature must order candidates the way geometry does."""
        actions = self.output["actions"]
        spatial = build_action_spatial_feature(actions, self.state)
        uav = self.state["uavs"][next(iter(self.state["uavs"]))]["position"]
        centers = {
            (item["field_id"], item["boundary_id"]): item["center"]
            for item in self.state["boundaries"]
        }
        truth = [
            hypot(
                centers[(action.field_id, action.entry_boundary_id)][0] - uav[0],
                centers[(action.field_id, action.entry_boundary_id)][1] - uav[1],
            )
            for action in actions
        ]
        by_feature = sorted(range(len(actions)), key=lambda i: float(spatial[i, 1]))
        by_truth = sorted(range(len(actions)), key=lambda i: truth[i])
        self.assertEqual(by_feature, by_truth)

    def test_policy_context_carries_the_spatial_block(self):
        embedded_dim = self.policy.action_embedding.output_dim
        expected = embedded_dim + ACTION_SPATIAL_DIM + self.policy.encoder.hidden_dim
        self.assertEqual(self.policy.action_decoder.network[0].in_features, expected)

        context = self.output["action_input"]
        self.assertEqual(tuple(context.shape), (len(self.output["actions"]), expected))
        block = context[:, embedded_dim:embedded_dim + ACTION_SPATIAL_DIM]
        expected_block = build_action_spatial_feature(self.output["actions"], self.state)
        self.assertTrue(torch.allclose(block.detach(), expected_block, atol=1e-6))

    def test_missing_geometry_degrades_to_zeros_instead_of_raising(self):
        feature = build_action_spatial_feature(self.output["actions"], None)
        self.assertEqual(tuple(feature.shape), (len(self.output["actions"]), ACTION_SPATIAL_DIM))
        self.assertTrue(torch.all(feature == 0.0))


if __name__ == "__main__":
    unittest.main()
