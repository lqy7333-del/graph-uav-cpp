"""Global state encoding shared by the actor and critic.

The global context ``z_G`` is built from three parts:

* **Graph context** ``g_graph`` — mean-pooled Field and Boundary embeddings
  ``concat(MeanPool(Z_F), MeanPool(Z_B))``.
* **UAV state** ``x_UAV`` — a fixed 10-dimensional capability/state vector.
* **Depot state** ``x_Depot`` — the normalized depot ``(x, y)`` position.

The three raw vectors are concatenated and projected by ``GlobalStateEncoder``.
Feature construction is kept as plain functions so it can be tested without a
full policy.
"""

from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_SCALES = {
    "position_scale_m": 10_000.0,
    "max_energy_scale": 30_000.0,
    "max_payload_scale": 100.0,
    "speed_scale": 20.0,
    "coverage_width_scale": 50.0,
}


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def _norm(value, scale):
    if scale <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, float(value) / scale))


def _uav_fields(uav_source) -> dict:
    """Extract capability/state fields from a ``UAV`` object or a state dict."""
    if isinstance(uav_source, dict):
        position = uav_source.get("position") or (0.0, 0.0)
        max_energy = float(uav_source.get("max_energy", 0.0))
        max_payload = float(uav_source.get("max_payload", 0.0))
        speed = float(uav_source.get("speed", 0.0))
        coverage_width = float(uav_source.get("coverage_width", 0.0))
        remaining_energy = float(uav_source.get("remaining_energy", 0.0))
        remaining_payload = float(uav_source.get("remaining_payload", max_payload))
    else:
        position = getattr(uav_source, "current_position", None) or (0.0, 0.0)
        max_energy = float(getattr(uav_source, "max_energy", 0.0))
        max_payload = float(getattr(uav_source, "max_payload", 0.0))
        speed = float(getattr(uav_source, "speed", 0.0))
        coverage_width = float(getattr(uav_source, "coverage_width", 0.0))
        remaining_energy = float(getattr(uav_source, "remaining_energy", 0.0))
        remaining_payload = float(getattr(uav_source, "remaining_payload", max_payload))
    energy_ratio = remaining_energy / max_energy if max_energy > 0.0 else 0.0
    payload_ratio = remaining_payload / max_payload if max_payload > 0.0 else 0.0
    return {
        "position": position,
        "max_energy": max_energy,
        "max_payload": max_payload,
        "speed": speed,
        "coverage_width": coverage_width,
        "remaining_energy": remaining_energy,
        "remaining_payload": remaining_payload,
        "energy_ratio": energy_ratio,
        "payload_ratio": payload_ratio,
    }


def build_uav_feature(uav_source, scales=None) -> torch.Tensor:
    """Return the 10-dim ``x_UAV`` feature vector.

    Order: ``[E_max, Q_max, speed, coverage_width, x_u, y_u,
    E_remaining, Q_remaining, energy_ratio, payload_ratio]``.
    ``E_remaining``/``Q_remaining`` are normalized by their global scales so
    they carry an *absolute* budget, while the trailing ratios carry the
    *fraction of own capacity* remaining.
    """
    scales = {**DEFAULT_SCALES, **(scales or {})}
    fields = _uav_fields(uav_source)
    x, y = fields["position"]
    return torch.tensor(
        [
            _norm(fields["max_energy"], scales["max_energy_scale"]),
            _norm(fields["max_payload"], scales["max_payload_scale"]),
            _norm(fields["speed"], scales["speed_scale"]),
            _norm(fields["coverage_width"], scales["coverage_width_scale"]),
            _norm(x, scales["position_scale_m"]),
            _norm(y, scales["position_scale_m"]),
            _norm(fields["remaining_energy"], scales["max_energy_scale"]),
            _norm(fields["remaining_payload"], scales["max_payload_scale"]),
            _clamp(fields["energy_ratio"]),
            _clamp(fields["payload_ratio"]),
        ],
        dtype=torch.float32,
    )


def build_depot_feature(depot_source, scales=None) -> torch.Tensor:
    """Return the 2-dim normalized ``x_Depot`` position vector."""
    scales = {**DEFAULT_SCALES, **(scales or {})}
    if isinstance(depot_source, dict):
        position = depot_source.get("position") or (0.0, 0.0)
    else:
        position = depot_source.get_position()
    x, y = position
    return torch.tensor(
        [
            _norm(x, scales["position_scale_m"]),
            _norm(y, scales["position_scale_m"]),
        ],
        dtype=torch.float32,
    )


def build_graph_context(field_embedding, boundary_embedding) -> torch.Tensor:
    """Return ``concat(MeanPool(Z_F), MeanPool(Z_B))`` with empty-safe pooling."""
    dim = field_embedding.shape[-1]
    field_mean = (
        field_embedding.mean(dim=0)
        if field_embedding.shape[0] > 0
        else field_embedding.new_zeros(dim)
    )
    boundary_mean = (
        boundary_embedding.mean(dim=0)
        if boundary_embedding.shape[0] > 0
        else boundary_embedding.new_zeros(dim)
    )
    return torch.cat([field_mean, boundary_mean], dim=-1)


class GlobalStateEncoder(nn.Module):
    """Project ``concat(g_graph, x_UAV, x_Depot)`` into ``z_G``."""

    def __init__(self, graph_dim: int, uav_dim: int = 10, depot_dim: int = 2,
                 hidden_dim: int = 128) -> None:
        super().__init__()
        self.graph_dim = int(graph_dim)
        self.uav_dim = int(uav_dim)
        self.depot_dim = int(depot_dim)
        self.input_dim = self.graph_dim + self.uav_dim + self.depot_dim
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

    def forward(self, graph_context, uav_feature, depot_feature):
        """Return the global context embedding ``z_G`` of shape ``[hidden_dim]``."""
        graph_context = graph_context.reshape(-1)
        uav_feature = uav_feature.reshape(-1)
        depot_feature = depot_feature.reshape(-1)
        x = torch.cat([graph_context, uav_feature, depot_feature], dim=-1)
        return self.network(x)
