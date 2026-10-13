"""Embedding of spatial policy actions with entry/exit role semantics."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionEmbedding(nn.Module):
    """Concatenate selected Field and role-tagged Boundary vectors.

    Entry and exit boundaries share the same Boundary embedding table; the two
    learnable role vectors ``r_entry``/``r_exit`` make the two slots carry
    distinct entry/exit semantics.
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = 3 * self.hidden_dim
        self.r_entry = nn.Parameter(torch.empty(self.hidden_dim))
        self.r_exit = nn.Parameter(torch.empty(self.hidden_dim))
        nn.init.normal_(self.r_entry, mean=0.0, std=0.1)
        nn.init.normal_(self.r_exit, mean=0.0, std=0.1)

    def forward(self, actions, field_embedding, boundary_embedding, graph):
        """Return ``[z_F, z_B(e_in)+r_entry, z_B(e_out)+r_exit]`` per candidate."""
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
        z_field = field_embedding[fi]
        z_entry = boundary_embedding[ei] + self.r_entry
        z_exit = boundary_embedding[xi] + self.r_exit
        return torch.cat((z_field, z_entry, z_exit), dim=-1)
