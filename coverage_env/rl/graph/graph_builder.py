"""Build PyTorch-Geometric-style heterogeneous Field/Boundary graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from torch_geometric.data import Data as _PyGData
except ImportError:  # Keep graph construction testable before optional deps install.
    _PyGData = None

from coverage_env.rl.graph.feature_builder import (
    BoundaryFeatureBuilder,
    FieldFeatureBuilder,
    GlobalFeatureBuilder,
)
from coverage_env.rl.graph.relation_builder import RelationFeatureBuilder


@dataclass
class _DataFallback:
    """Minimal Data-compatible container used only when PyG is unavailable."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def num_nodes(self):
        return int(self.x_field.shape[0] + self.x_boundary.shape[0])


class GraphBuilder:
    """Construct graph tensors and stable node mappings from Environment."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        config = config.get("rl", config)
        self.config = config.get("graph", config)
        self.field_features = FieldFeatureBuilder(
            max_area=float(self.config.get("max_area", 8_000.0)),
            max_perimeter=float(self.config.get("max_perimeter", 500.0)),
        )
        self.boundary_features = BoundaryFeatureBuilder(
            max_boundary_length=float(self.config.get("max_boundary_length", 200.0))
        )
        self.global_features = GlobalFeatureBuilder(
            position_scale_m=float(self.config.get("position_scale_m", 10_000.0))
        )
        self.relation_features = RelationFeatureBuilder(
            d_near=float(self.config.get("d_near_m", 1_500.0)),
            global_distance_scale=self.config.get("global_distance_scale_m"),
        )

    def build_graph(self, environment):
        """Return a PyG ``Data`` object with node features and typed relations.

        Field and Boundary feature matrices remain separate. ``edge_index``
        nevertheless uses a unified index space: Field nodes first, then
        Boundary nodes. The node mappings expose both local and unified IDs.
        """
        fields = list(environment.fields)
        field_id_to_index = {
            field.field_id: index for index, field in enumerate(fields)
        }
        boundary_id_to_index = {}
        boundary_key_to_index = {}
        boundary_records = []
        for field in fields:
            local = {}
            for boundary in field.boundaries:
                index = len(boundary_records)
                local[boundary.id] = index
                boundary_key_to_index[(field.field_id, boundary.id)] = index
                boundary_records.append((field, boundary))
            boundary_id_to_index[field.field_id] = local

        x_field = self.field_features.build(environment)
        x_boundary = self.boundary_features.build(environment, boundary_records)
        global_feature = self.global_features.build(environment)
        relations = self.relation_features.build(
            environment, field_id_to_index, boundary_id_to_index
        )

        data_type = _PyGData or _DataFallback
        data = data_type(
            x_field=x_field,
            x_boundary=x_boundary,
            edge_index=relations["edge_index"],
            edge_type=relations["edge_type"],
            edge_attr=relations["edge_attr"],
            global_feature=global_feature,
            relation_features=relations["relation_features"],
            relation_counts=relations["relation_counts"],
            relation_feature_dims=relations["relation_feature_dims"],
            field_id_to_index=field_id_to_index,
            boundary_id_to_index=boundary_id_to_index,
            boundary_key_to_index=boundary_key_to_index,
            field_node_index=dict(field_id_to_index),
            boundary_node_index={
                key: len(fields) + index
                for key, index in boundary_key_to_index.items()
            },
            field_ids=[field.field_id for field in fields],
            boundary_ids=[(field.field_id, boundary.id)
                          for field, boundary in boundary_records],
            num_nodes=len(fields) + len(boundary_records),
            graph_metadata={
                "edge_type_names": {
                    0: "field_boundary",
                    1: "boundary_boundary_intra",
                    2: "boundary_boundary_inter",
                },
                "feature_dims": {
                    "field": 5, "boundary": 2, "global": 5,
                    "relation": relations["relation_feature_dims"],
                    "edge_attr_padded": 4,
                },
                "d_near_m": self.relation_features.d_near,
                "uses_pyg_data": _PyGData is not None,
            },
        )
        return data


def build_graph(environment, config: dict | None = None):
    """Convenience entry point for one Environment graph snapshot."""
    return GraphBuilder(config).build_graph(environment)
