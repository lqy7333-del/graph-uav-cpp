"""The UAV graph node: zero-dilution agent-relative messages.

Design claims under test (see ``UavRelativeEncoder``):

1. The UAV is a real node: one node, ``|F| + |B|`` outgoing relation edges.
2. Those edges carry the relative offset *and* the normalized distance, so the
   graph computes "which candidate is closest" instead of leaving the decoder to
   infer it from raw coordinates.
3. Running them as their own stage is what keeps the message readable: every
   destination has exactly one incoming edge, so the destination softmax is
   exactly 1 and the message is not down-weighted against map edges.
4. The split of concerns holds: Stages 1-2 stay map-relative (unchanged when only
   the UAV moves) while the final embeddings become agent-relative.
"""

from __future__ import annotations

import unittest

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.encoder.hetero_encoder import CoverageGraphEncoder
from coverage_env.rl.graph.graph_builder import GraphBuilder
from coverage_env.rl.graph.relation_builder import EDGE_UB, EDGE_UF


class UavGraphNodeTests(unittest.TestCase):
    """Validate the UAV node, its relations, and the agent-relative stage."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG_PATH)
        cls.builder = GraphBuilder(cls.config["rl"])
        torch.manual_seed(0)
        cls.encoder = CoverageGraphEncoder.from_config(cls.config["rl"])
        cls.encoder.eval()

    def _graph_at(self, position):
        """Build the same instance with the UAV moved to ``position``."""
        environment = InstanceGenerator(self.config).generate(num_fields=5, seed=7)
        environment.reset()
        environment.uavs[0].current_position = position
        return self.builder.build_graph(environment)

    def test_uav_edges_cover_every_field_and_boundary(self):
        graph = self._graph_at((0.0, 0.0))
        field_count = graph.x_field.shape[0]
        boundary_count = graph.x_boundary.shape[0]
        self.assertEqual(graph.relation_counts[EDGE_UF], field_count)
        self.assertEqual(graph.relation_counts[EDGE_UB], boundary_count)
        self.assertEqual(graph.x_uav_node.shape, (1, 11))

    def test_uav_edge_distance_channel_is_well_scaled(self):
        """Guard the under-scaling failure mode: distances must span [0, 1]."""
        graph = self._graph_at((0.0, 0.0))
        mask = (graph.edge_type == EDGE_UF) | (graph.edge_type == EDGE_UB)
        distance = graph.edge_attr[mask][:, 2]
        self.assertTrue(torch.all(distance >= 0.0))
        self.assertTrue(torch.all(distance <= 1.0))
        self.assertGreater(float(distance.max()), 0.1)
        # The offset channels must carry real geometry too, not be all zero.
        self.assertGreater(float(graph.edge_attr[mask][:, :2].abs().max()), 0.1)

    def test_uav_message_attention_is_exactly_one(self):
        """Every destination has one incoming UAV edge, so nothing dilutes it."""
        graph = self._graph_at((0.0, 0.0))
        self.encoder(graph)
        attention = self.encoder.attention_debug["uav_relative"]["attention"]
        self.assertGreater(attention.numel(), 0)
        self.assertTrue(
            torch.all(attention == 1.0),
            f"UAV attention must be exactly 1, got {attention.unique().tolist()}",
        )

    def test_stages_one_and_two_stay_map_relative(self):
        """Moving only the UAV must not disturb the map-relative attention."""
        near, far = self._graph_at((0.0, 0.0)), self._graph_at((900.0, 600.0))
        self.encoder(near)
        map_attention = {
            key: self.encoder.attention_debug[key]["attention"].clone()
            for key in ("boundary_inter", "boundary_intra", "field_boundary")
        }
        far_output = self.encoder(far)
        for key, value in map_attention.items():
            self.assertTrue(
                torch.equal(value, self.encoder.attention_debug[key]["attention"]),
                f"{key} changed when only the UAV moved",
            )
        # ...while the final, agent-relative embeddings must respond to the move.
        near_output = self.encoder(near)
        self.assertGreater(
            float((far_output["field_embedding"] - near_output["field_embedding"]).abs().max()),
            1e-4,
        )
        self.assertGreater(
            float((far_output["boundary_embedding"] - near_output["boundary_embedding"]).abs().max()),
            1e-4,
        )

    def test_distance_channel_reaches_the_destination_embedding(self):
        """Editing *only* the UAV edge attributes must move the node embeddings.

        This is the mechanistic claim behind the whole design: the distance is an
        edge feature, so it enters the destination node's embedding through
        ``RelationGATLayer``'s message and attention, with no change to the
        topology or to any node feature.
        """
        base = self._graph_at((0.0, 0.0))
        moved = base.clone()
        mask = (moved.edge_type == EDGE_UF) | (moved.edge_type == EDGE_UB)
        attributes = moved.edge_attr.clone()
        attributes[mask] = attributes[mask] * 0.5
        moved.edge_attr = attributes

        self.assertTrue(torch.equal(base.edge_index, moved.edge_index))
        self.assertTrue(torch.equal(base.x_field, moved.x_field))
        self.assertTrue(torch.equal(base.x_boundary, moved.x_boundary))
        self.assertTrue(torch.equal(base.x_uav_node, moved.x_uav_node))

        before = self.encoder(base)["field_embedding"]
        after = self.encoder(moved)["field_embedding"]
        self.assertGreater(float((before - after).abs().max()), 1e-4)

    def test_uav_node_features_are_state_only(self):
        """No absolute position: the node vector must not move with the UAV."""
        near = self._graph_at((0.0, 0.0)).x_uav_node
        far = self._graph_at((900.0, 600.0)).x_uav_node
        self.assertTrue(torch.equal(near, far))


    def test_capability_scales_are_matched_to_uav_ranges(self):
        """Each capability normalizer must equal its ``uav.ranges`` upper bound.

        A scale far above the sampled range squashes the channel: at
        ``max_energy_scale=30000`` the 1200-1800 range landed in a 0.02-wide band,
        so "how big is this episode's battery" was nearly invisible next to
        ``payload_ratio`` (sd 0.006 vs 0.315). Taking the upper bound is the
        smallest scale that clips nothing, which maximizes the normalized span.
        """
        graph_config = self.config["rl"]["graph"]
        ranges = self.config["uav"]["ranges"]
        pairs = {
            "max_energy_scale": "max_energy",
            "max_payload_scale": "max_payload",
            "speed_scale": "speed",
            "coverage_width_scale": "coverage_width",
        }
        for scale_key, range_key in pairs.items():
            upper = float(ranges[range_key]["max"])
            scale = float(graph_config[scale_key])
            self.assertGreaterEqual(
                scale, upper, f"{scale_key} would clip the sampled range"
            )
            self.assertLessEqual(
                scale, 1.5 * upper, f"{scale_key} under-conditions its channel"
            )

    def test_uav_scales_agree_between_graph_and_global_features(self):
        """The graph's UAV node and the z_G uav vector must share one scale."""
        from coverage_env.rl.decoder.global_state_encoder import DEFAULT_SCALES

        graph_config = self.config["rl"]["graph"]
        for key in (
            "max_energy_scale", "max_payload_scale", "speed_scale",
            "coverage_width_scale",
        ):
            self.assertAlmostEqual(
                float(graph_config[key]), float(DEFAULT_SCALES[key]), places=6
            )

    def test_consumption_rates_are_observable(self):
        """alpha/beta/payload_per_area are randomised per episode, so they must be
        visible. Before this they were neither in ``x_uav`` nor in the UAV node, so
        the agent could not see how fast it burns energy or payload."""
        environment = InstanceGenerator(self.config).generate(seed=42)
        rates = []
        for _ in range(12):
            environment.reset()
            uav = environment.uavs[0]
            node = self.builder.build_graph(environment).x_uav_node[0]
            rates.append((uav.alpha, uav.beta, uav.payload_per_area))
            graph_scale = self.config["rl"]["graph"]
            self.assertAlmostEqual(
                float(node[6]), uav.alpha / float(graph_scale["alpha_scale"]), places=5
            )
            self.assertAlmostEqual(
                float(node[7]), uav.beta / float(graph_scale["beta_scale"]), places=5
            )
            self.assertAlmostEqual(
                float(node[8]),
                uav.payload_per_area / float(graph_scale["payload_per_area_scale"]),
                places=5,
            )
        values = torch.tensor(rates)
        self.assertGreater(float(values.std(dim=0).min()), 1e-4)
        self.assertGreater(float(values[:, 2].std()), 1e-4)

    def test_capability_channels_are_not_squashed(self):
        """Across randomised episodes the capability channels must span a real band."""
        environment = InstanceGenerator(self.config).generate(seed=42)
        rows = []
        for _ in range(12):
            environment.reset()
            rows.append(self.builder.build_graph(environment).x_uav_node[0])
        stacked = torch.stack(rows)
        # Dims 0/1 are the *state* ratios, which are 1.0 at reset by construction;
        # dims 2-8 are the per-episode capability/consumption channels this scale fix
        # targets. Dims 9/10 are the per-parcel look-ahead (work still outstanding and
        # how many loads it needs): both are constant at reset, so they are excluded
        # rather than silently widening the failure to a channel this test is not about.
        capability = stacked[:, 2:9]
        spread = capability.max(dim=0).values - capability.min(dim=0).values
        self.assertGreater(
            float(spread.min()), 0.1, f"a capability channel is squashed: {spread.tolist()}"
        )


if __name__ == "__main__":
    unittest.main()
