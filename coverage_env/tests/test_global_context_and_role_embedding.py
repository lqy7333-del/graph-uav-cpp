"""Tests for global-state encoding and entry/exit role embeddings."""

from __future__ import annotations

import unittest

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.decoder.global_state_encoder import (
    build_site_feature,
    build_uav_feature,
)


class GlobalContextAndRoleEmbeddingTests(unittest.TestCase):
    """Verify capability-aware global context and boundary role semantics."""

    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG_PATH)
        self.environment = InstanceGenerator(self.config).generate(
            num_fields=5, seed=20261012
        )
        self.environment.reset()
        self.policy = CoveragePolicy(self.environment, config=self.config)
        self.policy.eval()

    def test_global_embedding_depends_on_uav_capabilities(self):
        env = self.environment
        base = self.policy(env.get_state(), environment=env)["global_embedding"]
        uav = env.uavs[0]
        saved = {
            name: getattr(uav, name)
            for name in ("max_energy", "max_payload", "speed", "coverage_width")
        }
        try:
            uav.max_energy = 1000.0
            uav.max_payload = 10.0
            uav.speed = 5.0
            uav.coverage_width = 3.0
            altered = self.policy(env.get_state(), environment=env)["global_embedding"]
        finally:
            for name, value in saved.items():
                setattr(uav, name, value)
        self.assertFalse(torch.allclose(base, altered, atol=1e-6))

    def test_entry_exit_role_embeddings_differ(self):
        output = self.policy(
            self.environment.get_state(), environment=self.environment
        )
        boundary_embedding = output["encoder_output"]["boundary_embedding"]
        r_entry = self.policy.action_embedding.r_entry
        r_exit = self.policy.action_embedding.r_exit
        row = boundary_embedding[0]
        self.assertFalse(torch.allclose(row + r_entry, row + r_exit))

    def test_entry_exit_share_boundary_encoder(self):
        embedding = self.policy.action_embedding
        self.assertEqual(embedding.r_entry.shape, (self.policy.encoder.hidden_dim,))
        self.assertEqual(embedding.r_exit.shape, (self.policy.encoder.hidden_dim,))
        # Only the two role vectors are learnable; there is no separate entry/exit
        # projection, so both slots draw from the same shared boundary embedding.
        self.assertEqual(len(list(embedding.parameters())), 2)
        self.assertFalse(any(
            isinstance(module, torch.nn.Linear) for module in embedding.modules()
        ))

    def test_uav_feature_is_13d(self):
        x_uav = build_uav_feature(self.environment.uavs[0])
        self.assertEqual(tuple(x_uav.shape), (13,))

    def test_site_feature_is_12d_and_env_equals_state(self):
        environment = self.environment
        environment.reset()
        state = environment.get_state()
        from_environment = build_site_feature(environment)
        from_state = build_site_feature(state)
        self.assertEqual(tuple(from_environment.shape), (12,))
        self.assertTrue(torch.equal(from_environment, from_state))

    def test_site_feature_is_permutation_invariant_but_instance_sensitive(self):
        environment = InstanceGenerator(self.config).generate(seed=5)
        state = environment.reset()
        baseline = build_site_feature(state)
        reordered = dict(state)
        reordered["fields"] = dict(reversed(list(state["fields"].items())))
        self.assertTrue(torch.allclose(baseline, build_site_feature(reordered)))

        other = InstanceGenerator(self.config).generate(seed=6)
        other.reset()
        self.assertFalse(torch.allclose(baseline, build_site_feature(other)))

    def test_global_embedding_depends_on_site_layout(self):
        """Pooled embeddings ignore field *positions*, so x_site must carry them.

        Translating every field by the same offset leaves the graph embeddings
        and the UAV/depot features untouched, so any change in ``z_G`` can only
        come from the layout feature.
        """
        environment = self.environment
        state = environment.reset()
        graph = self.policy.graph_builder.build_graph(environment)
        shifted = dict(state)
        shifted["fields"] = {
            field_id: {
                **field, "center": (field["center"][0] + 500.0, field["center"][1])
            }
            for field_id, field in state["fields"].items()
        }
        self.assertFalse(
            torch.allclose(build_site_feature(state), build_site_feature(shifted)),
            "translating the fields must change the layout feature",
        )
        with torch.no_grad():
            baseline = self.policy.encode_graph(
                graph, environment=environment, state=state
            )[1]
            after = self.policy.encode_graph(
                graph, environment=environment, state=shifted
            )[1]
        self.assertFalse(torch.allclose(baseline, after, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
