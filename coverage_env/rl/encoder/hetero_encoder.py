"""Two-stage relation-aware encoder for Field/Boundary coverage graphs."""

from __future__ import annotations

import torch
import torch.nn as nn

from coverage_env.rl.graph.relation_builder import (
    EDGE_BB_INTER,
    EDGE_BB_INTRA,
    EDGE_FB,
    EDGE_UB,
    EDGE_UF,
)
from coverage_env.rl.encoder.feature_embedding import FeatureEmbedding
from coverage_env.rl.encoder.relation_gat import RelationGATLayer


class BoundaryInterEncoder(nn.Module):
    """Stage 1: update Boundary embeddings using cross-field relations only."""

    def __init__(self, hidden_dim=128, relation_dim=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.layer = RelationGATLayer(
            hidden_dim, relation_dim, num_heads=num_heads, dropout=dropout
        )

    def forward(self, h_boundary, edge_index, edge_attr):
        """Return Stage 1 embeddings and attention for E_BB_inter."""
        return self.layer(h_boundary, edge_index, edge_attr, num_nodes=h_boundary.shape[0])


class BoundaryIntraEncoder(nn.Module):
    """Stage 2a: update Boundary embeddings using same-field relations."""

    def __init__(self, hidden_dim=128, relation_dim=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.layer = RelationGATLayer(
            hidden_dim, relation_dim, num_heads=num_heads, dropout=dropout
        )

    def forward(self, h_boundary, edge_index, edge_attr):
        """Return intra-coverage Boundary embeddings and attention weights."""
        return self.layer(h_boundary, edge_index, edge_attr, num_nodes=h_boundary.shape[0])


class FieldBoundaryEncoder(nn.Module):
    """Stage 2b: exchange coverage-structure messages across E_FB."""

    def __init__(self, hidden_dim=128, relation_dim=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.layer = RelationGATLayer(
            hidden_dim, relation_dim, num_heads=num_heads, dropout=dropout
        )

    def forward(self, h_field, h_boundary, edge_index, edge_attr):
        """Update Fields and Boundaries using bidirectional FB messages.

        Phase 9 stores one Field -> owned Boundary edge. The reverse edge is
        added here only for message passing so that both node types can update;
        no Field-Field relation is created.
        """
        n_field = h_field.shape[0]
        joined = torch.cat([h_field, h_boundary], dim=0)
        bidirectional_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        bidirectional_attr = torch.cat([edge_attr, edge_attr], dim=0)
        updated, attention = self.layer(
            joined, bidirectional_index, bidirectional_attr,
            num_nodes=joined.shape[0],
        )
        return updated[:n_field], updated[n_field:], attention, bidirectional_index


class UavRelativeEncoder(nn.Module):
    """Stage 3: turn map-relative embeddings into agent-relative ones.

    Stages 1-2 describe the map (which fields are adjacent, how a field is
    structured) and are deliberately invariant to where the agent is. This stage
    adds the missing half of the decision: for every Field and every Boundary,
    "where is the UAV relative to me". The relative offset and the normalized
    distance travel as *edge features*, so ``RelationGATLayer`` uses them both in
    the attention score and in the message content -- the graph computes the
    distance instead of the decoder having to infer it.

    Running as its own layer is what keeps that message readable: every
    destination has exactly one incoming edge here, so ``_destination_softmax``
    returns exactly 1 and the message arrives undiluted. Folding these edges into
    Stage 1 would place them in the same per-destination softmax as every
    ``E_BB_inter`` edge, down-weighting them to roughly ``1/(K+1)`` for a Boundary
    with ``K`` inter-field neighbours.

    The layer is residual, so the Stage 1-2 embeddings are preserved and merely
    augmented rather than replaced.
    """

    def __init__(self, hidden_dim=128, relation_dim=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.layer = RelationGATLayer(
            hidden_dim, relation_dim, num_heads=num_heads, dropout=dropout
        )

    def forward(self, h_field, h_boundary, h_uav, edge_index, edge_attr):
        """Return updated Field/Boundary/UAV embeddings and UAV-edge attention."""
        n_field = h_field.shape[0]
        n_boundary = h_boundary.shape[0]
        joined = torch.cat([h_field, h_boundary, h_uav], dim=0)
        updated, attention = self.layer(
            joined, edge_index, edge_attr, num_nodes=joined.shape[0]
        )
        return (
            updated[:n_field],
            updated[n_field:n_field + n_boundary],
            updated[n_field + n_boundary:],
            attention,
        )


class CoverageGraphEncoder(nn.Module):
    """Encode a graph into Field and Boundary embeddings.

    Stage 1 models transitions between fields through Boundary nodes. Stage 2
    models each field's coverage structure through intra-boundary and
    Field-Boundary relations. Stage 3 (``UavRelativeEncoder``) then conditions the
    result on the UAV's pose, producing the agent-relative embeddings that the
    action decoder reads. Attention weights from every stage are retained in
    ``attention_debug`` for inspection.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
        field_input_dim: int = 10,
        boundary_input_dim: int = 5,
        uav_input_dim: int = 11,
        relation_input_dim: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.feature_embedding = FeatureEmbedding(
            field_input_dim=field_input_dim,
            boundary_input_dim=boundary_input_dim,
            uav_input_dim=uav_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.boundary_inter_encoder = BoundaryInterEncoder(
            hidden_dim, relation_input_dim, num_heads, dropout
        )
        self.boundary_intra_encoder = BoundaryIntraEncoder(
            hidden_dim, relation_input_dim, num_heads, dropout
        )
        self.field_boundary_encoder = FieldBoundaryEncoder(
            hidden_dim, relation_input_dim, num_heads, dropout
        )
        self.uav_relative_encoder = UavRelativeEncoder(
            hidden_dim, relation_input_dim, num_heads, dropout
        )
        self.attention_debug: dict = {}

    @classmethod
    def from_config(cls, config: dict):
        """Create an encoder from the ``rl.encoder`` YAML section."""
        config = config.get("rl", config)
        values = config.get("encoder", config)
        return cls(
            hidden_dim=int(values.get("hidden_dim", 128)),
            num_heads=int(values.get("num_heads", 4)),
            dropout=float(values.get("dropout", 0.0)),
        )

    def forward(self, graph):
        """Return ``field_embedding``, ``boundary_embedding``, and global input."""
        x_field = graph.x_field
        x_boundary = graph.x_boundary
        x_uav_node = graph.x_uav_node
        edge_index = graph.edge_index.to(x_field.device)
        edge_type = graph.edge_type.to(x_field.device)
        edge_attr = graph.edge_attr.to(x_field.device)
        n_field = x_field.shape[0]
        n_boundary = x_boundary.shape[0]

        h_field_0, h_boundary_0, h_uav_0 = self.feature_embedding(
            x_field, x_boundary, x_uav_node
        )

        inter_mask = edge_type == EDGE_BB_INTER
        inter_edges = edge_index[:, inter_mask] - n_field
        inter_attr = edge_attr[inter_mask]
        h_boundary_1, inter_attention = self.boundary_inter_encoder(
            h_boundary_0, inter_edges, inter_attr
        )

        intra_mask = edge_type == EDGE_BB_INTRA
        intra_edges = edge_index[:, intra_mask] - n_field
        intra_attr = edge_attr[intra_mask]
        h_boundary_intra, intra_attention = self.boundary_intra_encoder(
            h_boundary_1, intra_edges, intra_attr
        )

        fb_mask = edge_type == EDGE_FB
        fb_edges = edge_index[:, fb_mask]
        fb_attr = edge_attr[fb_mask]
        h_field_1, h_boundary_2, fb_attention, fb_bidirectional_edges = (
            self.field_boundary_encoder(
                h_field_0, h_boundary_intra, fb_edges, fb_attr
            )
        )

        # Stage 3: agent-relative readout. Only the UAV relation families feed
        # this layer, so each destination sees exactly one incoming edge and the
        # distance message cannot be diluted by map edges.
        uav_mask = (edge_type == EDGE_UF) | (edge_type == EDGE_UB)
        uav_edges = edge_index[:, uav_mask]
        uav_attr = edge_attr[uav_mask]
        h_field_2, h_boundary_3, _h_uav_2, uav_attention = self.uav_relative_encoder(
            h_field_1, h_boundary_2, h_uav_0, uav_edges, uav_attr
        )

        self.attention_debug = {
            "boundary_inter": {
                "edge_index": inter_edges.detach(),
                "edge_attr": inter_attr.detach(),
                "attention": inter_attention.detach(),
                "edge_type": EDGE_BB_INTER,
            },
            "boundary_intra": {
                "edge_index": intra_edges.detach(),
                "edge_attr": intra_attr.detach(),
                "attention": intra_attention.detach(),
                "edge_type": EDGE_BB_INTRA,
            },
            "field_boundary": {
                "edge_index": fb_bidirectional_edges.detach(),
                "edge_attr": torch.cat([fb_attr, fb_attr], dim=0).detach(),
                "attention": fb_attention.detach(),
                "edge_type": EDGE_FB,
            },
            "uav_relative": {
                "edge_index": uav_edges.detach(),
                "edge_attr": uav_attr.detach(),
                "attention": uav_attention.detach(),
                "edge_type": (EDGE_UF, EDGE_UB),
            },
        }
        return {
            "field_embedding": h_field_2,
            "boundary_embedding": h_boundary_3,
            "global_feature": graph.global_feature,
        }

    def top_inter_neighbors(self, graph, field_id, boundary_id, limit=5):
        """Return highest-attention incoming inter-field neighbors for a node.

        Each row contains ``neighbor_id`` as ``(field_id, boundary_id)`` and
        the mean attention over heads. A Boundary without incoming inter edges
        returns an empty list.
        """
        if "boundary_inter" not in self.attention_debug:
            raise RuntimeError("Call the encoder before requesting attention debug data.")
        local_target = graph.boundary_id_to_index[field_id][boundary_id]
        debug = self.attention_debug["boundary_inter"]
        edges = debug["edge_index"].detach().cpu()
        attention = debug["attention"].detach().mean(dim=1).cpu()
        neighbors = []
        for edge_position in range(edges.shape[1]):
            source, target = int(edges[0, edge_position]), int(edges[1, edge_position])
            if target != local_target:
                continue
            neighbors.append({
                "neighbor_id": graph.boundary_ids[source],
                "attention": float(attention[edge_position]),
            })
        return sorted(neighbors, key=lambda item: item["attention"], reverse=True)[:limit]
