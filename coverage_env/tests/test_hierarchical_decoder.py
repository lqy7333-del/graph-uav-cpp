"""Factored decode: ``p(field) * p(entry|field) * p(exit|field,entry)``.

The chain rule makes this exact, not an approximation: the joint over candidates
still sums to one and its entropy equals the chain-rule sum, so PPO's log-prob and
entropy term are unchanged. These tests pin both properties.
"""

from __future__ import annotations

import unittest

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy


class HierarchicalDecodeTests(unittest.TestCase):
    """The factored distribution must be a proper distribution, exactly."""

    def setUp(self):
        torch.manual_seed(0)
        self.config = load_config(DEFAULT_CONFIG_PATH)
        self.config["rl"]["decoder"]["hierarchical_decoding"] = True
        self.environment = InstanceGenerator(self.config).generate(
            num_fields=5, seed=1005, randomize_each_reset=False
        )
        self.policy = CoveragePolicy(self.environment, config=self.config)
        self.policy.eval()
        self.state = self.environment.reset()
        with torch.no_grad():
            self.output = self.policy(self.state, environment=self.environment)

    def test_joint_distribution_is_normalised(self):
        distribution = self.output["action_distribution"]
        self.assertAlmostEqual(float(distribution.sum()), 1.0, places=5)
        self.assertTrue(bool((distribution[self.output["mask"]] > 0).all()))
        self.assertTrue(bool((distribution[~self.output["mask"]] == 0).all()))

    def test_field_probabilities_cover_the_feasible_parcels(self):
        probabilities = self.output["hierarchical"]["field_probabilities"]
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=5)
        self.assertGreater(int(probabilities.numel()), 1)
        # A single parcel must not swallow the whole level (the bug this guards
        # against was a per-parcel segment softmax, which returns exactly 1 each).
        self.assertLess(float(probabilities.max()), 0.999)

    def test_joint_equals_product_of_levels(self):
        hierarchical = self.output["hierarchical"]
        segments = hierarchical["field_segments"]
        expected = (
            hierarchical["field_probabilities"][segments]
            * hierarchical["entry_probabilities"]
            * hierarchical["exit_probabilities"]
        )
        actual = self.output["action_distribution"]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_entropy_matches_the_chain_rule(self):
        """H(joint) must equal H(F) + H(E|F) + H(X|F,E)."""
        hierarchical = self.output["hierarchical"]
        distribution = self.output["action_distribution"]
        joint_entropy = float(
            -(distribution * torch.log(distribution.clamp_min(1e-12))).sum()
        )

        field = hierarchical["field_probabilities"]
        field_segments = hierarchical["field_segments"]
        field_entropy = float(-(field * torch.log(field.clamp_min(1e-12))).sum())

        # Conditional entropies weighted by the joint -- the exact chain rule,
        # which is why PPO's entropy term needs no re-tuning.
        with torch.no_grad():
            entry_groups = hierarchical["entry_probabilities"]
            exit_groups = hierarchical["exit_probabilities"]
            entry_entropy = float(
                -(
                    distribution
                    * torch.log(entry_groups.clamp_min(1e-12))
                ).sum()
            )
            exit_entropy = float(
                -(
                    distribution
                    * torch.log(exit_groups.clamp_min(1e-12))
                ).sum()
            )
        total = field_entropy + entry_entropy + exit_entropy
        self.assertAlmostEqual(joint_entropy, total, places=4)

    def test_batched_and_single_graph_paths_agree(self):
        context = {
            "observation": self.state,
            "graph": self.output["graph"],
            "actions": self.output["actions"],
            "mask": self.output["mask"],
        }
        with torch.no_grad():
            single = self.policy(self.state, environment=self.environment)
            batched = self.policy.forward_batched([context])[0]
        self.assertTrue(torch.allclose(
            single["action_distribution"],
            batched["action_distribution"],
            atol=1e-6,
        ))

    def test_flat_decoder_still_available(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        config["rl"]["decoder"]["hierarchical_decoding"] = False
        policy = CoveragePolicy(self.environment, config=config)
        policy.eval()
        with torch.no_grad():
            output = policy(self.state, environment=self.environment)
        self.assertNotIn("hierarchical", output)
        self.assertAlmostEqual(float(output["action_distribution"].sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
