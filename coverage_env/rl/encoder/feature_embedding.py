"""Initial MLP projections for Field, Boundary, and UAV node features."""

from __future__ import annotations

import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """Project each node type's raw features into one shared hidden space."""

    def __init__(
        self,
        field_input_dim: int = 7,
        boundary_input_dim: int = 4,
        uav_input_dim: int = 9,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(field_input_dim, boundary_input_dim, uav_input_dim, hidden_dim) <= 0:
            raise ValueError("Feature dimensions and hidden_dim must be positive.")
        self.field_mlp = self._make_mlp(field_input_dim, hidden_dim, dropout)
        self.boundary_mlp = self._make_mlp(boundary_input_dim, hidden_dim, dropout)
        # The UAV node carries agent state only; its geometry travels on the
        # UAV->Field / UAV->Boundary edges, not in this vector.
        self.uav_mlp = self._make_mlp(uav_input_dim, hidden_dim, dropout)

    @staticmethod
    def _make_mlp(input_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
        """Create a small normalized two-layer feature projection."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x_field, x_boundary, x_uav_node):
        """Return initial embeddings ``h_F0, h_B0, h_U0``."""
        return (
            self.field_mlp(x_field),
            self.boundary_mlp(x_boundary),
            self.uav_mlp(x_uav_node),
        )
