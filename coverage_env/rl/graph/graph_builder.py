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
    UavNodeFeatureBuilder,
)
from coverage_env.rl.graph.relation_builder import (
    EDGE_BB_INTER,
    EDGE_BB_INTRA,
    EDGE_FB,
    EDGE_UB,
    EDGE_UF,
    RelationFeatureBuilder,
)


@dataclass
class _DataFallback:
    """Minimal Data-compatible container used only when PyG is unavailable."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def num_nodes(self):
        return int(
            self.x_field.shape[0] + self.x_boundary.shape[0] + self.x_uav_node.shape[0]
        )


class GraphBuilder:
    """Construct graph tensors and stable node mappings from Environment."""

    def __init__(self, config: dict | None = None) -> None:
        config = config or {}
        config = config.get("rl", config)
        self.config = config.get("graph", config)
        self.field_features = FieldFeatureBuilder(
            max_area=float(self.config.get("max_area", 8_000.0)),
            max_perimeter=float(self.config.get("max_perimeter", 500.0)),
            position_scale_m=float(self.config.get("position_scale_m", 800.0)),
        )
        self.boundary_features = BoundaryFeatureBuilder(
            max_boundary_length=float(self.config.get("max_boundary_length", 200.0)),
            position_scale_m=float(self.config.get("position_scale_m", 800.0)),
        )
        self.global_features = GlobalFeatureBuilder(
            position_scale_m=float(self.config.get("position_scale_m", 800.0))
        )
        self.uav_features = UavNodeFeatureBuilder(
            max_energy_scale=float(self.config.get("max_energy_scale", 1_800.0)),
            max_payload_scale=float(self.config.get("max_payload_scale", 60.0)),
            speed_scale=float(self.config.get("speed_scale", 10.0)),
            coverage_width_scale=float(self.config.get("coverage_width_scale", 12.0)),
        )
        self.relation_features = RelationFeatureBuilder(
            d_near=float(self.config.get("d_near_m", 1_500.0)),
            global_distance_scale=self.config.get("global_distance_scale_m"),
            uav_distance_scale=float(
                self.config.get("uav_distance_scale_m", 1_200.0)
            ),
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
        x_uav_node = self.uav_features.build(environment)
        global_feature = self.global_features.build(environment)
        relations = self.relation_features.build(
            environment, field_id_to_index, boundary_id_to_index
        )
        # Unified node space: Field nodes, then Boundary nodes, then the single
        # UAV node. Appending the UAV last keeps every existing index expression
        # (``edge_index - n_field`` for boundary rows) unchanged.
        uav_node_index = len(fields) + len(boundary_records)

        data_type = _PyGData or _DataFallback
        data = data_type(
            x_field=x_field,
            x_boundary=x_boundary,
            x_uav_node=x_uav_node,
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
            uav_node_index=uav_node_index,
            field_ids=[field.field_id for field in fields],
            boundary_ids=[(field.field_id, boundary.id)
                          for field, boundary in boundary_records],
            num_nodes=len(fields) + len(boundary_records) + int(x_uav_node.shape[0]),
            graph_metadata={
                "edge_type_names": {
                    EDGE_FB: "field_boundary",
                    EDGE_BB_INTRA: "boundary_boundary_intra",
                    EDGE_BB_INTER: "boundary_boundary_inter",
                    EDGE_UF: "uav_field",
                    EDGE_UB: "uav_boundary",
                },
                "feature_dims": {
                    "field": 7, "boundary": 4, "global": 5,
                    "uav": int(x_uav_node.shape[1]),
                    "relation": relations["relation_feature_dims"],
                    "edge_attr_padded": 4,
                },
                "d_near_m": self.relation_features.d_near,
                "uav_distance_scale_m": self.relation_features.uav_distance_scale,
                "uses_pyg_data": _PyGData is not None,
            },
        )
        return data


def build_graph(environment, config: dict | None = None):
    """Convenience entry point for one Environment graph snapshot."""
    return GraphBuilder(config).build_graph(environment)
