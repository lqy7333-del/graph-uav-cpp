"""Embedding of spatial policy actions and global context."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    """Concatenate selected Field, entry Boundary, and exit Boundary vectors."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = 3 * self.hidden_dim

    def forward(self, actions, field_embedding, boundary_embedding, graph):
        """Return candidate vectors with shape ``[N, 3 * hidden_dim]``."""
        if not actions:
            return field_embedding.new_empty((0, self.output_dim))
        field_indices, entry_indices, exit_indices = [], [], []
        for action in actions:
            try:
                field_indices.append(graph.field_id_to_index[action.field_id])
                local = graph.boundary_id_to_index[action.field_id]
                entry_indices.append(local[action.entry_boundary_id])
                exit_indices.append(local[action.exit_boundary_id])
            except KeyError as exc:
                raise ValueError(f"Action references an unknown field/boundary: {action}") from exc
        device = field_embedding.device
        fi = torch.tensor(field_indices, dtype=torch.long, device=device)
        ei = torch.tensor(entry_indices, dtype=torch.long, device=device)
        xi = torch.tensor(exit_indices, dtype=torch.long, device=device)
        return torch.cat((field_embedding[fi], boundary_embedding[ei],
                          boundary_embedding[xi]), dim=-1)


class GlobalEncoder(nn.Module):
    """Project UAV and depot features to the action-context dimension."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )

    def forward(self, global_feature):
        """Return a global context vector of shape ``[hidden_dim]``."""
        return self.network(global_feature.reshape(1, -1)).squeeze(0)
