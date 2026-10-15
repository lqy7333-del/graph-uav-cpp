"""Global state encoding shared by the actor and critic.

The global context ``z_G`` is built from four parts:

* **Graph context** ``g_graph`` — mean-pooled Field and Boundary embeddings
  ``concat(MeanPool(Z_F), MeanPool(Z_B))``.
* **UAV state** ``x_UAV`` — a fixed 10-dimensional capability/state vector.
* **Depot state** ``x_Depot`` — the normalized depot ``(x, y)`` position.
* **Site layout** ``x_site`` — a 12-dimensional summary of the field layout
  (count, total area, depot->field and field->field distance statistics, area
  spread). Pooled embeddings are invariant to the number, order, and position of
  fields, so without this the value function cannot see the geometry that the
  path-length reward depends on.

The four raw vectors are concatenated and projected by ``GlobalStateEncoder``.
Feature construction is kept as plain functions so it can be tested without a
full policy, and each accepts either an ``Environment`` or a serialized state
dict so PPO replay uses the transition's own values.
"""

from __future__ import annotations

from math import hypot

import torch
import torch.nn as nn


DEFAULT_SCALES = {
    "position_scale_m": 10_000.0,
    "max_energy_scale": 30_000.0,
    "max_payload_scale": 100.0,
    "speed_scale": 20.0,
    "coverage_width_scale": 50.0,
    # Site-layout scales. The episode reward is path length per unit covered
    # area, so it depends on *where the fields are* relative to the depot and to
    # each other; these bound the matching normalized layout features.
    "max_fields_scale": 20.0,
    "max_total_area_scale": 120_000.0,
    "site_distance_scale_m": 2_000.0,
    "max_field_area_scale": 6_000.0,
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


def _site_tables(source):
    """Return ``(areas, centroids, depot_position)`` from an env or a state dict."""
    if isinstance(source, dict):
        fields = list(source.get("fields", {}).values())
        centroids = [tuple(field.get("center") or (0.0, 0.0)) for field in fields]
        areas = [float(field.get("area", 0.0)) for field in fields]
        depot = source.get("depot") or {}
        position = depot.get("position") or (0.0, 0.0)
        return areas, centroids, (float(position[0]), float(position[1]))
    centroids = [field.centroid for field in source.fields]
    areas = [float(field.area) for field in source.fields]
    position = source.depot.get_position()
    return areas, centroids, (float(position[0]), float(position[1]))


def build_site_feature(source, scales=None) -> torch.Tensor:
    """Return the 12-dim site-layout feature vector ``x_site``.

    Order: ``[num_fields, total_area, depot_dist_{mean,max,min,std},
    pair_dist_{mean,max,min,std}, area_{mean,std}]``.

    Mean-pooled graph embeddings are invariant to the *number, order, and
    position* of fields, yet the episode reward (path length per unit covered
    area) depends on exactly the layout: how far the depot is and how spread out
    the fields are. A probe on real episodes showed ``z_G`` recovers only
    R^2 = 0.14 of the return while these explicit layout scalars recover 0.54
    (ceiling 0.86). The same function serves an ``Environment`` and a serialized
    state dict, so PPO replay sees the transition's own layout.
    """
    scales = {**DEFAULT_SCALES, **(scales or {})}
    areas, centroids, depot = _site_tables(source)
    depot_distances = [hypot(cx - depot[0], cy - depot[1]) for cx, cy in centroids]
    pair_distances = [
        hypot(cx - ox, cy - oy)
        for index, (cx, cy) in enumerate(centroids)
        for ox, oy in centroids[index + 1:]
    ]

    def stats(values):
        if not values:
            return 0.0, 0.0, 0.0, 0.0
        mean = sum(values) / len(values)
        spread = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        return mean, max(values), min(values), spread

    depot_mean, depot_max, depot_min, depot_std = stats(depot_distances)
    pair_mean, pair_max, pair_min, pair_std = stats(pair_distances)
    area_mean, _, _, area_std = stats(areas)

    distance_scale = max(float(scales["site_distance_scale_m"]), 1e-9)
    area_scale = max(float(scales["max_field_area_scale"]), 1e-9)
    total_area_scale = max(float(scales["max_total_area_scale"]), 1e-9)
    fields_scale = max(float(scales["max_fields_scale"]), 1e-9)

    return torch.tensor([
        _clamp(len(centroids) / fields_scale),
        _clamp(sum(areas) / total_area_scale),
        _clamp(depot_mean / distance_scale),
        _clamp(depot_max / distance_scale),
        _clamp(depot_min / distance_scale),
        _clamp(depot_std / distance_scale),
        _clamp(pair_mean / distance_scale),
        _clamp(pair_max / distance_scale),
        _clamp(pair_min / distance_scale),
        _clamp(pair_std / distance_scale),
        _clamp(area_mean / area_scale),
        _clamp(area_std / area_scale),
    ], dtype=torch.float32)


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
    """Project ``concat(g_graph, x_UAV, x_Depot, x_site)`` into ``z_G``."""

    def __init__(self, graph_dim: int, uav_dim: int = 10, depot_dim: int = 2,
                 site_dim: int = 12, hidden_dim: int = 128) -> None:
        super().__init__()
        self.graph_dim = int(graph_dim)
        self.uav_dim = int(uav_dim)
        self.depot_dim = int(depot_dim)
        self.site_dim = int(site_dim)
        self.input_dim = (
            self.graph_dim + self.uav_dim + self.depot_dim + self.site_dim
        )
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

    def _assemble(self, graph_context, uav_feature, depot_feature, site_feature):
        """Concatenate the parts along the last dim, in a fixed order."""
        parts = [graph_context, uav_feature, depot_feature]
        if self.site_dim > 0:
            if site_feature is None:
                raise ValueError("site_feature is required when site_dim > 0.")
            parts.append(site_feature)
        return torch.cat(parts, dim=-1)

    def forward(self, graph_context, uav_feature, depot_feature, site_feature=None):
        """Return the global context embedding ``z_G`` of shape ``[hidden_dim]``."""
        return self.network(self._assemble(
            graph_context.reshape(-1),
            uav_feature.reshape(-1),
            depot_feature.reshape(-1),
            site_feature.reshape(-1) if site_feature is not None else None,
        ))

    def forward_batched(
        self, graph_context, uav_feature, depot_feature, site_feature=None
    ):
        """Return ``z_G`` for a graph batch, shape ``[num_graphs, hidden_dim]``.

        ``forward`` flattens its inputs because it serves exactly one graph; this
        variant keeps the leading graph dimension so a disjoint-union batch can
        share one projection pass. Every feature is ``[G, dim]``.
        """
        def _rows(feature):
            if feature is not None and feature.ndim == 1:
                return feature.unsqueeze(0)
            return feature

        return self.network(self._assemble(
            _rows(graph_context), _rows(uav_feature),
            _rows(depot_feature), _rows(site_feature),
        ))
