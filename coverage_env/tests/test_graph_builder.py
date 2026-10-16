"""Tests for the Field/Boundary heterogeneous graph representation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib
import torch
from torch_geometric.data import Data

matplotlib.use("Agg")

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.graph.graph_builder import GraphBuilder
from coverage_env.rl.graph.relation_builder import (
    EDGE_BB_INTER,
    EDGE_BB_INTRA,
    EDGE_FB,
    EDGE_UB,
    EDGE_UF,
)
from coverage_env.visualization.plot import plot_graph


class GraphBuilderTests(unittest.TestCase):
    """Validate graph node types, relation counts, dimensions, and near edges."""

    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG_PATH)
        self.environment = InstanceGenerator(self.config).generate(
            num_fields=5, seed=20261012
        )
        self.builder = GraphBuilder(self.config["rl"])
        self.graph = self.builder.build_graph(self.environment)

    def test_graph_has_field_boundary_and_uav_nodes(self):
        graph = self.graph
        self.assertIsInstance(graph, Data)
        field_count = len(self.environment.fields)
        boundary_count = sum(len(field.boundaries) for field in self.environment.fields)
        self.assertEqual(graph.x_field.shape, (field_count, 7))
        self.assertEqual(graph.x_boundary.shape, (boundary_count, 4))
        self.assertTrue(torch.all(graph.x_field[:, 5:7].abs() <= 1.0))
        self.assertTrue(torch.all(graph.x_boundary[:, 2:4].abs() <= 1.0))
        position_scale = self.config["rl"]["graph"]["position_scale_m"]
        for index, field in enumerate(self.environment.fields):
            expected_center = torch.tensor(field.centroid) / position_scale
            self.assertTrue(torch.allclose(graph.x_field[index, 5:7], expected_center))
        boundary_records = [
            boundary for field in self.environment.fields for boundary in field.boundaries
        ]
        for index, boundary in enumerate(boundary_records):
            expected_center = torch.tensor(boundary.center) / position_scale
            self.assertTrue(torch.allclose(graph.x_boundary[index, 2:4], expected_center))
        self.assertEqual(graph.global_feature.shape, (5,))
        # One UAV node is appended after every Field and Boundary node.
        self.assertEqual(graph.x_uav_node.shape, (1, 6))
        self.assertEqual(graph.uav_node_index, field_count + boundary_count)
        self.assertEqual(graph.num_nodes, field_count + boundary_count + 1)
        self.assertTrue(torch.all(graph.x_uav_node >= 0.0))
        self.assertTrue(torch.all(graph.x_uav_node <= 1.0))
        self.assertEqual(graph.relation_counts[EDGE_UF], field_count)
        self.assertEqual(graph.relation_counts[EDGE_UB], boundary_count)
        # Every UAV edge starts at the UAV node.
        uav_mask = (graph.edge_type == EDGE_UF) | (graph.edge_type == EDGE_UB)
        self.assertTrue(torch.all(
            graph.edge_index[0, uav_mask] == graph.uav_node_index
        ))
        for relation in (EDGE_UF, EDGE_UB):
            self.assertEqual(graph.relation_features[relation].shape[1], 3)
        self.assertEqual(graph.relation_counts[EDGE_FB], boundary_count)
        self.assertEqual(
            graph.relation_counts[EDGE_BB_INTRA],
            sum(len(field.boundaries) * (len(field.boundaries) - 1)
                for field in self.environment.fields),
        )
        self.assertEqual(graph.edge_attr.shape[1], 4)
        self.assertEqual(graph.relation_features[EDGE_FB].shape[1], 4)
        self.assertEqual(graph.relation_features[EDGE_BB_INTRA].shape[1], 3)
        self.assertEqual(graph.relation_features[EDGE_BB_INTER].shape[1], 3)
        self.assertEqual(len(graph.field_id_to_index), field_count)
        self.assertEqual(len(graph.boundary_key_to_index), boundary_count)

        # FB edges must go Field -> Boundary; BB edges must have Boundary
        # endpoints; UAV edges must start at the UAV node and target a Field or a
        # Boundary (the UAV -> Field family is the one that legitimately targets a
        # Field node, so it is checked separately from the map relations).
        uav_index = field_count + boundary_count
        for (source, target), edge_type in zip(
            graph.edge_index.t().tolist(), graph.edge_type.tolist()
        ):
            if edge_type == EDGE_FB:
                self.assertLess(source, field_count)
                self.assertGreaterEqual(target, field_count)
            elif edge_type in (EDGE_UF, EDGE_UB):
                self.assertEqual(source, uav_index)
                self.assertNotEqual(target, uav_index)
            else:
                self.assertGreaterEqual(source, field_count)
                self.assertGreaterEqual(target, field_count)
        self.assertNotIn("field_field", graph.graph_metadata["edge_type_names"].values())

    def test_position_channels_are_well_scaled(self):
        """Node coordinates must not be drowned out by their sibling features.

        Measured failure mode: with a 10000 m scale the coordinate channels sat at
        |mean| 0.045 while area/perimeter/compactness/remaining were 0.38-1.0, and
        the policy never learned the nearest-field decision. Guard against that
        regression rather than the exact constant.
        """
        coordinates = self.graph.x_field[:, 5:7]
        siblings = self.graph.x_field[:, :5]
        self.assertGreater(float(coordinates.abs().mean()), 0.1)
        self.assertGreater(
            float(coordinates.abs().mean()), 0.2 * float(siblings.abs().mean())
        )
        # Coordinate channels must actually vary, otherwise "which field is
        # nearest" is unrepresentable.
        self.assertGreater(float(coordinates.std(dim=0).min()), 0.01)
        self.assertTrue(torch.all(self.graph.x_boundary[:, 2:4].abs() <= 1.0))

    def test_graph_and_global_position_scales_agree(self):
        """Node centres and the z_G uav/depot position must share one frame."""
        from coverage_env.rl.decoder.global_state_encoder import DEFAULT_SCALES

        self.assertAlmostEqual(
            float(self.config["rl"]["graph"]["position_scale_m"]),
            float(DEFAULT_SCALES["position_scale_m"]),
            places=6,
        )

    def test_inter_edges_obey_near_threshold_and_visualization(self):
        graph = self.graph
        near_limit = self.builder.relation_features.d_near
        node_fields = {}
        for field in self.environment.fields:
            for boundary in field.boundaries:
                node_index = graph.boundary_node_index[(field.field_id, boundary.id)]
                node_fields[node_index] = (field.field_id, boundary)
        inter_edges = [
            (source, target)
            for (source, target), edge_type in zip(
                graph.edge_index.t().tolist(), graph.edge_type.tolist()
            )
            if edge_type == EDGE_BB_INTER
        ]
        for source, target in inter_edges:
            source_field, source_boundary = node_fields[source]
            target_field, target_boundary = node_fields[target]
            self.assertNotEqual(source_field, target_field)
            dx = source_boundary.center[0] - target_boundary.center[0]
            dy = source_boundary.center[1] - target_boundary.center[1]
            self.assertLess((dx * dx + dy * dy) ** 0.5, near_limit)

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory) / "graph.png"
            figure, _ = plot_graph(
                self.environment, graph, show=False, save_path=str(output)
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)
            matplotlib.pyplot.close(figure)

    def test_build_graph_prints_instance_summary(self):
        graph = self.graph
        print(
            "Graph instance: "
            f"fields={graph.x_field.shape[0]}, "
            f"boundaries={graph.x_boundary.shape[0]}, "
            f"E_FB={graph.relation_counts[EDGE_FB]}, "
            f"E_BB_intra={graph.relation_counts[EDGE_BB_INTRA]}, "
            f"E_BB_inter={graph.relation_counts[EDGE_BB_INTER]}, "
            f"feature_dims={graph.graph_metadata['feature_dims']}"
        )


if __name__ == "__main__":
    unittest.main()
