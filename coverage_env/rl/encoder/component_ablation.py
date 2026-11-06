"""Parameter-matched node-local replacements for C/D boundary interaction ablations."""
from __future__ import annotations
import torch
from torch import nn


class NodeLocalResidual(nn.Module):
    """A residual per-node transform that ignores edges and all other nodes.

    It keeps a residual path, LayerNorm and GELU structure similar to the
    replaced RelationGAT layer. No neighbour aggregation or edge attributes are
    read. Hidden width is chosen only from the replaced layer's parameter count.
    """
    def __init__(self, hidden_dim: int, reference_parameter_count: int,
                 num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        # 2*H*W + 3*H + W parameters; width is the unique closest integer.
        base = 3 * hidden_dim
        self.width = max(1, round((int(reference_parameter_count) - base)
                                   / float(2 * hidden_dim + 1)))
        self.input = nn.Linear(hidden_dim, self.width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(self.width, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, node_features, edge_index, edge_attr):
        del edge_index, edge_attr
        local = self.output(self.dropout(self.activation(self.input(node_features))))
        updated = self.normalization(node_features + local)
        empty_attention = node_features.new_empty((0, self.num_heads))
        return updated, empty_attention


def reference_gat_parameter_count(hidden_dim: int, relation_dim: int, num_heads: int) -> int:
    """Exact parameter count of RelationGATLayer used by the replaced layer."""
    H, R, K = int(hidden_dim), int(relation_dim), int(num_heads)
    # source/target projections (no bias)
    n = 2 * H * H * K
    # relation embedding: R->H (bias) + H->H*K (bias)
    n += H * R + H + H * H * K + H * K
    # three attention vectors
    n += 3 * H * K
    # output/residual projections (no bias) + LayerNorm
    n += 2 * H * H + 2 * H
    return int(n)


def component_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
