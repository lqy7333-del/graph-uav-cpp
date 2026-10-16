"""Tests for the two-stage relation-aware CoverageGraphEncoder."""

from __future__ import annotations

import copy
import unittest

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.encoder.hetero_encoder import CoverageGraphEncoder
from coverage_env.rl.graph.graph_builder import GraphBuilder
from coverage_env.rl.graph.relation_builder import EDGE_BB_INTER, EDGE_BB_INTRA, EDGE_FB


class CoverageGraphEncoderTests(unittest.TestCase):
    """Check embedding shapes, stage attention, and relation feature effects."""

    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG_PATH)
        cls.environment = InstanceGenerator(cls.config).generate(
            num_fields=5, seed=20261012
        )
        cls.graph = GraphBuilder(cls.config["rl"]).build_graph(cls.environment)
        cls.encoder = CoverageGraphEncoder.from_config(cls.config["rl"])
        cls.encoder.eval()

    def test_forward_shapes_and_stage_attention(self):
        output = self.encoder(self.graph)
        self.assertEqual(output["field_embedding"].shape, (5, 128))
        self.assertEqual(output["boundary_embedding"].shape,
                         (self.graph.x_boundary.shape[0], 128))
        self.assertTrue(torch.equal(output["global_feature"], self.graph.global_feature))
        self.assertEqual(
            set(self.encoder.attention_debug),
            {"boundary_inter", "boundary_intra", "field_boundary", "uav_relative"},
        )
        self.assertEqual(
            self.encoder.attention_debug["boundary_inter"]["edge_type"],
            EDGE_BB_INTER,
        )
        self.assertEqual(
            self.encoder.attention_debug["boundary_intra"]["edge_type"],
            EDGE_BB_INTRA,
        )
        self.assertEqual(
            self.encoder.attention_debug["field_boundary"]["edge_type"], EDGE_FB
        )

        # Incoming attention weights sum to one independently for every head.
        edge_index = self.encoder.attention_debug["boundary_inter"]["edge_index"]
        weights = self.encoder.attention_debug["boundary_inter"]["attention"]
        if edge_index.shape[1]:
            target = int(edge_index[1, 0])
            selected = edge_index[1] == target
            self.assertTrue(torch.allclose(
                weights[selected].sum(dim=0),
                torch.ones(weights.shape[1]), atol=1e-5,
            ))

    def test_attention_debug_lists_cross_field_neighbors(self):
        self.encoder(self.graph)
        debug = self.encoder.attention_debug["boundary_inter"]
        edges = debug["edge_index"]
        target = int(edges[1, 0])
        field_id, boundary_id = self.graph.boundary_ids[target]
        neighbors = self.encoder.top_inter_neighbors(
            self.graph, field_id, boundary_id, limit=3
        )
        self.assertTrue(neighbors)
        self.assertLessEqual(len(neighbors), 3)
        self.assertTrue(all(item["neighbor_id"][0] != field_id for item in neighbors))
        self.assertTrue(all(0.0 <= item["attention"] <= 1.0 for item in neighbors))

    def test_relation_features_participate_in_encoder(self):
        self.encoder.eval()
        baseline = self.encoder(self.graph)["boundary_embedding"]
        altered = copy.deepcopy(self.graph)
        inter = altered.edge_type == EDGE_BB_INTER
        altered.edge_attr[inter, 0] += 0.4
        changed = self.encoder(altered)["boundary_embedding"]
        self.assertFalse(torch.allclose(baseline, changed))


if __name__ == "__main__":
    unittest.main()
