"""Relation-aware multi-head graph attention message passing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationGATLayer(nn.Module):
    """Attention layer whose scores and messages use edge relation features.

    For each directed edge ``i -> j``, the score is computed from transformed
    source node, destination node, and an MLP embedding of the edge feature.
    The message also includes the relation embedding, so edge attributes affect
    both neighbor weighting and the information passed to the destination.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        relation_dim: int = 4,
        num_heads: int = 4,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or relation_dim <= 0 or num_heads <= 0:
            raise ValueError("hidden_dim, relation_dim, and num_heads must be positive.")
        self.hidden_dim = hidden_dim
        self.relation_dim = relation_dim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)

        projected_dim = num_heads * hidden_dim
        self.source_projection = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.target_projection = nn.Linear(hidden_dim, projected_dim, bias=False)
        self.relation_embedding = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projected_dim),
        )
        self.attention_source = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.attention_target = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.attention_relation = nn.Parameter(torch.empty(num_heads, hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.residual_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the attention vectors and projection layers."""
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)
        nn.init.xavier_uniform_(self.attention_relation)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.ones_(self.normalization.weight)
        nn.init.zeros_(self.normalization.bias)

    def forward(self, node_features, edge_index, edge_attr, *, num_nodes=None):
        """Return updated node features and ``[edge, head]`` attention weights."""
        if node_features.ndim != 2 or node_features.shape[1] != self.hidden_dim:
            raise ValueError("node_features must have shape [N, hidden_dim].")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if edge_attr.ndim != 2 or edge_attr.shape[1] != self.relation_dim:
            raise ValueError("edge_attr must have shape [E, relation_dim].")
        if edge_index.shape[1] != edge_attr.shape[0]:
            raise ValueError("edge_index and edge_attr must have matching edge counts.")

        n_nodes = int(node_features.shape[0] if num_nodes is None else num_nodes)
        if edge_index.shape[1] == 0:
            empty_attention = node_features.new_empty((0, self.num_heads))
            output = self.normalization(self.residual_projection(node_features))
            return F.elu(output), empty_attention

        source, target = edge_index[0], edge_index[1]
        source_h = self.source_projection(node_features[source]).view(
            -1, self.num_heads, self.hidden_dim
        )
        target_h = self.target_projection(node_features[target]).view(
            -1, self.num_heads, self.hidden_dim
        )
        relation_h = self.relation_embedding(edge_attr).view(
            -1, self.num_heads, self.hidden_dim
        )
        score = (
            (source_h * self.attention_source).sum(dim=-1)
            + (target_h * self.attention_target).sum(dim=-1)
            + (relation_h * self.attention_relation).sum(dim=-1)
        )
        score = F.leaky_relu(score, negative_slope=self.negative_slope)
        attention = self._destination_softmax(score, target, n_nodes)
        dropped_attention = F.dropout(attention, p=self.dropout, training=self.training)

        messages = source_h + relation_h
        weighted_messages = messages * dropped_attention.unsqueeze(-1)
        aggregate = node_features.new_zeros((n_nodes, self.num_heads, self.hidden_dim))
        aggregate.index_add_(0, target, weighted_messages)
        aggregate = aggregate.mean(dim=1)
        output = self.output_projection(aggregate) + self.residual_projection(node_features)
        output = self.normalization(output)
        return F.elu(output), attention

    @staticmethod
    def _destination_softmax(scores, destination, num_nodes):
        """Compute stable independent softmaxes over incoming edges per head."""
        heads = scores.shape[1]
        indices = destination.unsqueeze(-1).expand(-1, heads)
        maxima = scores.new_full((num_nodes, heads), float("-inf"))
        maxima.scatter_reduce_(0, indices, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - maxima[destination])
        denominator = scores.new_zeros((num_nodes, heads))
        denominator.index_add_(0, destination, exp_scores)
        return exp_scores / denominator[destination].clamp_min(1e-12)
