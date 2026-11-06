"""Relation-free, parameter-matched CHANet ablation.

Each node sees only its own features and its graph's UAV state. No edges or
other field/boundary nodes enter the encoder. Global pooling remains in policy.
"""
import torch
from torch import nn
from .feature_embedding import FeatureEmbedding


class NodeMLPEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        from .hetero_encoder import CoverageGraphEncoder
        self.hidden_dim = hidden_dim
        # Count the A encoder without advancing the experiment RNG.
        with torch.random.fork_rng(devices=[]):
            reference = CoverageGraphEncoder(hidden_dim, num_heads, dropout)
        self.reference_parameters = sum(p.numel() for p in reference.parameters())
        self.feature_embedding = FeatureEmbedding(hidden_dim=hidden_dim, dropout=dropout)
        base = sum(p.numel() for p in self.feature_embedding.parameters())
        # Six active residual blocks: each (2H -> W -> H) + LayerNorm(H).
        self.width = max(1, round((self.reference_parameters-base-18*hidden_dim)
                                 / (6*(3*hidden_dim+1))))
        def blocks():
            return nn.ModuleList([nn.Sequential(
                nn.Linear(2*hidden_dim, self.width), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(self.width, hidden_dim), nn.LayerNorm(hidden_dim),
            ) for _ in range(3)])
        self.field_blocks, self.boundary_blocks = blocks(), blocks()
        self.attention_debug = {}

    def forward(self, graph):
        f, b, u = self.feature_embedding(graph.x_field, graph.x_boundary, graph.x_uav_node)
        if hasattr(graph, 'field_batch'):
            if not bool(((graph.uav_offsets[1:]-graph.uav_offsets[:-1]) == 1).all()):
                raise ValueError('node_mlp ablation requires one UAV per graph')
            uf, ub = u[graph.field_batch], u[graph.boundary_batch]
        else:
            if u.shape[0] != 1:
                raise ValueError('node_mlp ablation requires one UAV')
            uf, ub = u.expand(f.shape[0], -1), u.expand(b.shape[0], -1)
        for layer in self.field_blocks:
            f = f + layer(torch.cat((f, uf), dim=-1))
        for layer in self.boundary_blocks:
            b = b + layer(torch.cat((b, ub), dim=-1))
        return dict(field_embedding=f, boundary_embedding=b, global_feature=graph.global_feature)
