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
        self.assertEqual(graph.x_uav_node.shape, (1, 6))

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


if __name__ == "__main__":
    unittest.main()
