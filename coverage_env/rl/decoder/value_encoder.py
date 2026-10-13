"""PPO-ready state representation pooling module (no value head)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueEncoder(nn.Module):
    """Mean-pool typed node embeddings and global context into state features."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, field_embedding, boundary_embedding, global_embedding):
        """Return pooled state embedding with shape ``[1, hidden_dim]``."""
        hidden_dim = global_embedding.shape[-1]
        field_mean = self._mean_or_zero(field_embedding, hidden_dim)
        boundary_mean = self._mean_or_zero(boundary_embedding, hidden_dim)
        global_row = global_embedding.reshape(1, hidden_dim)
        pooled = torch.cat([field_mean, boundary_mean, global_row], dim=-1)
        return self.network(pooled)

    @staticmethod
    def _mean_or_zero(values, hidden_dim):
        if values.shape[0] == 0:
            return values.new_zeros((1, hidden_dim))
        return values.mean(dim=0, keepdim=True)
