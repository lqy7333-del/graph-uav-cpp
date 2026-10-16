"""Prototype: disjoint-union batching for the heterogeneous coverage encoder.

**This module is not wired into the training path.** It exists so the "图 batch
化" direction from ``docs/training_performance_notes.md`` (P1) can be validated
numerically before any integration:

* ``collate_graphs`` concatenates several single-graph objects into one
  disjoint union (fields first, then boundaries) and records the node/edge
  ownership and per-graph offsets.
* ``BatchedCoverageGraphEncoder.forward_batched`` runs the *same* submodules as
  ``CoverageGraphEncoder.forward`` once over the union. The type-separated node
  layout (all fields, then all boundaries) keeps every attention index identical
  to the single-graph expression, so the only substitution is per-graph pooled
  context in place of the single global mean.

Because no edge crosses graph boundaries, every per-node and per-graph result
must match the existing per-graph path. ``coverage_env.tests.
test_batched_encoder_equivalence`` asserts exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from coverage_env.rl.encoder.hetero_encoder import CoverageGraphEncoder
from coverage_env.rl.graph.relation_builder import (
    EDGE_BB_INTER,
    EDGE_BB_INTRA,
    EDGE_FB,
    EDGE_UB,
    EDGE_UF,
)


@dataclass
class BatchedGraph:
    """A disjoint union of single-graph objects in one unified index space.

    Node space is *type separated*: every Field node first, then every Boundary
    node, then every UAV node. A graph ``k`` owns field rows
    ``[field_offsets[k], field_offsets[k+1])``, boundary rows
    ``[boundary_offsets[k], boundary_offsets[k+1])`` and uav rows
    ``[uav_offsets[k], uav_offsets[k+1])``. Unified node ids follow the
    single-graph convention: field ``i`` of graph ``k`` is
    ``field_offsets[k] + i``, boundary ``j`` is
    ``num_fields + boundary_offsets[k] + j``, and uav ``m`` is
    ``num_fields + num_boundaries + uav_offsets[k] + m``.
    """

    x_field: torch.Tensor
    x_boundary: torch.Tensor
    x_uav_node: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_attr: torch.Tensor
    global_feature: torch.Tensor
    field_batch: torch.Tensor
    boundary_batch: torch.Tensor
    uav_batch: torch.Tensor
    edge_batch: torch.Tensor
    field_offsets: torch.Tensor
    boundary_offsets: torch.Tensor
    uav_offsets: torch.Tensor
    num_fields: int
    num_boundaries: int
    num_uavs: int
    num_graphs: int

    @property
    def num_nodes(self) -> int:
        """Return the total number of Field + Boundary + UAV nodes."""
        return int(self.num_fields + self.num_boundaries + self.num_uavs)

    def to(self, device) -> "BatchedGraph":
        """Return a copy of this batch with every tensor moved to ``device``."""
        moved = {
            name: getattr(self, name).to(device)
            for name in (
                "x_field", "x_boundary", "x_uav_node", "edge_index", "edge_type",
                "edge_attr", "global_feature", "field_batch", "boundary_batch",
                "uav_batch", "edge_batch", "field_offsets", "boundary_offsets",
                "uav_offsets",
            )
        }
        return BatchedGraph(
            **moved,
            num_fields=self.num_fields,
            num_boundaries=self.num_boundaries,
            num_uavs=self.num_uavs,
            num_graphs=self.num_graphs,
        )


def _cumulative_offsets(counts: list[int], dtype: torch.dtype) -> torch.Tensor:
    """Return ``[0, c0, c0+c1, ...]`` as a tensor for scatter/slicing math."""
    values = [0]
    for count in counts:
        values.append(values[-1] + int(count))
    return torch.tensor(values, dtype=dtype)


def _repeat_index(counts: list[int], dtype: torch.dtype) -> torch.Tensor:
    """Return ``[0]*c0 + [1]*c1 + ...`` mapping each row to its graph index."""
    return torch.repeat_interleave(
        torch.arange(len(counts), dtype=dtype),
        torch.tensor(counts, dtype=dtype),
    )


def collate_graphs(graphs) -> BatchedGraph:
    """Concatenate single-graph objects into one disjoint-union ``BatchedGraph``.

    ``graphs`` may be PyG ``Data`` objects or the project's ``_DataFallback``;
    only ``x_field``/``x_boundary``/``edge_index``/``edge_type``/``edge_attr``/
    ``global_feature`` are read. Edge endpoints are remapped from each graph's
    own unified space into the batch's unified space, so no edge ever crosses a
    component boundary.
    """
    graph_list = list(graphs)
    if not graph_list:
        raise ValueError("collate_graphs requires at least one graph.")

    field_counts = [int(graph.x_field.shape[0]) for graph in graph_list]
    boundary_counts = [int(graph.x_boundary.shape[0]) for graph in graph_list]
    uav_counts = [int(graph.x_uav_node.shape[0]) for graph in graph_list]
    edge_dtype = graph_list[0].edge_index.dtype
    field_offsets = _cumulative_offsets(field_counts, edge_dtype)
    boundary_offsets = _cumulative_offsets(boundary_counts, edge_dtype)
    uav_offsets = _cumulative_offsets(uav_counts, edge_dtype)
    total_fields = int(field_offsets[-1].item())
    total_boundaries = int(boundary_offsets[-1].item())

    edges, edge_types, edge_attrs, edge_batches = [], [], [], []
    for graph_index, graph in enumerate(graph_list):
        graph_edges = graph.edge_index
        graph_field_count = field_counts[graph_index]
        graph_boundary_count = boundary_counts[graph_index]
        # A single graph numbers fields first, then boundaries, then the UAV.
        # Each type moves into its own block of the batch's shared space, so the
        # three endpoint kinds need three distinct remappings.
        is_field = graph_edges < graph_field_count
        is_boundary = graph_edges < graph_field_count + graph_boundary_count
        remapped = torch.where(
            is_field,
            graph_edges + field_offsets[graph_index],
            torch.where(
                is_boundary,
                graph_edges - graph_field_count + total_fields
                + boundary_offsets[graph_index],
                graph_edges - graph_field_count - graph_boundary_count
                + total_fields + total_boundaries + uav_offsets[graph_index],
            ),
        )
        edges.append(remapped)
        edge_types.append(graph.edge_type)
        edge_attrs.append(graph.edge_attr)
        edge_batches.append(torch.full(
            (graph_edges.shape[1],), graph_index, dtype=edge_dtype,
        ))

    return BatchedGraph(
        x_field=torch.cat([graph.x_field for graph in graph_list], dim=0),
        x_boundary=torch.cat([graph.x_boundary for graph in graph_list], dim=0),
        x_uav_node=torch.cat([graph.x_uav_node for graph in graph_list], dim=0),
        edge_index=torch.cat(edges, dim=1),
        edge_type=torch.cat(edge_types, dim=0),
        edge_attr=torch.cat(edge_attrs, dim=0),
        global_feature=torch.stack(
            [graph.global_feature.reshape(-1) for graph in graph_list], dim=0
        ),
        field_batch=_repeat_index(field_counts, torch.long),
        boundary_batch=_repeat_index(boundary_counts, torch.long),
        uav_batch=_repeat_index(uav_counts, torch.long),
        edge_batch=torch.cat(edge_batches, dim=0),
        field_offsets=field_offsets,
        boundary_offsets=boundary_offsets,
        uav_offsets=uav_offsets,
        num_fields=total_fields,
        num_boundaries=total_boundaries,
        num_uavs=int(uav_offsets[-1].item()),
        num_graphs=len(graph_list),
    )


def segment_mean(values: torch.Tensor, index: torch.Tensor, groups: int) -> torch.Tensor:
    """Return the per-group mean of ``values`` rows (empty groups become zero).

    This is the batched replacement for the single-graph ``mean(dim=0)``; it is
    the only place where batching must be aware of graph ownership.
    """
    if values.shape[0] == 0:
        return values.new_zeros((groups, values.shape[-1]))
    counts = values.new_zeros((groups,), dtype=values.dtype)
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    totals = values.new_zeros((groups, values.shape[-1]))
    totals.index_add_(0, index, values)
    return totals / counts.clamp_min(1.0).unsqueeze(-1)


def batched_graph_context(
    field_embedding: torch.Tensor,
    boundary_embedding: torch.Tensor,
    batch: BatchedGraph,
) -> torch.Tensor:
    """Return ``concat(MeanPool(Z_F), MeanPool(Z_B))`` per graph, shape ``[G, 2H]``."""
    field_mean = segment_mean(field_embedding, batch.field_batch, batch.num_graphs)
    boundary_mean = segment_mean(
        boundary_embedding, batch.boundary_batch, batch.num_graphs
    )
    return torch.cat([field_mean, boundary_mean], dim=-1)


def encode_graph_batch(encoder, batch: BatchedGraph) -> dict:
    """Run ``encoder``'s submodules once over a disjoint-union ``BatchedGraph``.

    ``encoder`` is any ``CoverageGraphEncoder`` instance, so ``CoveragePolicy``
    can batch *its own* encoder in place — no wrapping, copying, or subclassing.

    Only **one** thing actually differs from the single-graph path: global mean
    pooling of the Field/Boundary embeddings becomes a per-graph
    ``segment_mean``. Every attention index is unchanged, because the
    type-separated batch layout (all fields first, then all boundaries) keeps
    the single-graph convention intact: a boundary row is still reached by
    ``unified_node_id - n_field_total``, and ``FieldBoundaryEncoder`` already
    splits ``cat([field, boundary])`` at the total field count. Message passing
    therefore never crosses a component, and the destination softmax stays
    per-node.
    """
    x_field = batch.x_field
    x_boundary = batch.x_boundary
    x_uav_node = batch.x_uav_node
    edge_index = batch.edge_index
    edge_type = batch.edge_type
    edge_attr = batch.edge_attr
    n_field_total = x_field.shape[0]

    h_field_0, h_boundary_0, h_uav_0 = encoder.feature_embedding(
        x_field, x_boundary, x_uav_node
    )

    # Boundary encoders consume the *concatenated* boundary matrix, so their
    # indices must stay globally unique across graphs. Subtracting the total
    # field count is therefore identical to the single-graph expression
    # ``edge_index - n_field``; no per-graph base offset belongs here.
    inter_mask = edge_type == EDGE_BB_INTER
    inter_edges = edge_index[:, inter_mask] - n_field_total
    inter_attr = edge_attr[inter_mask]
    h_boundary_1, inter_attention = encoder.boundary_inter_encoder(
        h_boundary_0, inter_edges, inter_attr
    )

    intra_mask = edge_type == EDGE_BB_INTRA
    intra_edges = edge_index[:, intra_mask] - n_field_total
    intra_attr = edge_attr[intra_mask]
    h_boundary_intra, intra_attention = encoder.boundary_intra_encoder(
        h_boundary_1, intra_edges, intra_attr
    )

    # FieldBoundaryEncoder already assumes "all fields, then all boundaries"
    # and unified FB edges, so the type-separated batch layout needs no change.
    fb_mask = edge_type == EDGE_FB
    fb_edges = edge_index[:, fb_mask]
    fb_attr = edge_attr[fb_mask]
    h_field_1, h_boundary_2, fb_attention, fb_bidirectional_edges = (
        encoder.field_boundary_encoder(
            h_field_0, h_boundary_intra, fb_edges, fb_attr
        )
    )

    # Stage 3: the UAV relation families only, in the same shared index space.
    # Every destination again has exactly one incoming edge, so no per-graph
    # bookkeeping is needed beyond the endpoints collate_graphs already remapped.
    uav_mask = (edge_type == EDGE_UF) | (edge_type == EDGE_UB)
    uav_edges = edge_index[:, uav_mask]
    uav_attr = edge_attr[uav_mask]
    h_field_2, h_boundary_3, _h_uav_2, uav_attention = encoder.uav_relative_encoder(
        h_field_1, h_boundary_2, h_uav_0, uav_edges, uav_attr
    )

    return {
        "field_embedding": h_field_2,
        "boundary_embedding": h_boundary_3,
        "global_feature": batch.global_feature,
        "graph_context": batched_graph_context(h_field_2, h_boundary_3, batch),
        "attention": {
            "boundary_inter": inter_attention,
            "boundary_intra": intra_attention,
            "field_boundary": fb_attention,
            "uav_relative": uav_attention,
        },
        "attended_edges": {
            "boundary_inter": inter_edges,
            "boundary_intra": intra_edges,
            "field_boundary": fb_bidirectional_edges,
            "uav_relative": uav_edges,
        },
        "batch": batch,
    }


class BatchedCoverageGraphEncoder(CoverageGraphEncoder):
    """Thin subclass exposing the batch entry point as a module method.

    ``forward`` (single graph) is inherited unchanged, so the same instance can
    serve both paths; ``forward_batched`` delegates to :func:`encode_graph_batch`.
    """

    def forward_batched(self, batch: BatchedGraph) -> dict:
        """Encode a disjoint-union batch and return batched embeddings + context."""
        return encode_graph_batch(self, batch)


__all__ = [
    "BatchedGraph",
    "BatchedCoverageGraphEncoder",
    "batched_graph_context",
    "collate_graphs",
    "encode_graph_batch",
    "segment_mean",
]
