"""Per-action spatial features: the distances needed to pick a field.

The graph gives the actor no way to tell which candidate field is closest. Field
node features are ``[area, perimeter, compactness, orientation, remaining]`` and
boundary features are ``[length, orientation]`` -- neither carries a position --
while every edge feature is relative to a field's *own* centroid, so the whole
graph is translation invariant. The only absolute position reaches the decoder as
``[uav_x, uav_y]`` inside the global context, which is broadcast identically to
every candidate action.

Measured consequence: the trained policy chose the nearest unfinished field 18.2%
of the time against an 11.7% random baseline (untrained: 19.1%), travelling 2.2x
the necessary distance, while a nearest-field heuristic scored -34.50 against the
policy's -43.76 on the same instances. These features make "go to the nearest
field" a linear function of the policy's input.
"""

from __future__ import annotations

from math import hypot

import torch

ACTION_SPATIAL_DIM = 5

# Distances are metres; the workspace spans a few kilometres, matching the
# ``site_distance_scale_m`` used by the global site summary.
DEFAULT_SCALES = {"distance_scale_m": 2_000.0}


def _point(value):
    """Return ``(x, y)`` as floats, or ``None`` when the value is unusable."""
    if value is None:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError, IndexError):
        return None


def _distance(first, second):
    """Euclidean distance, or ``None`` when either endpoint is unknown."""
    if first is None or second is None:
        return None
    return hypot(first[0] - second[0], first[1] - second[1])


def _norm(value, scale: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value) / scale))


def _tables(source):
    """Return ``(uav_xy, depot_xy, field_centers, boundary_centers)``.

    Accepts either an ``Environment`` or a serialized ``state`` dict so the same
    features can be produced on the rollout path and when replaying a stored
    transition.
    """
    if source is None:
        return None, None, {}, {}

    if isinstance(source, dict):
        uavs = source.get("uavs") or {}
        uav = next(iter(uavs.values()), None)
        uav_xy = _point((uav or {}).get("position"))
        depot_xy = _point((source.get("depot") or {}).get("position"))
        field_centers = {
            field_id: _point(item.get("center"))
            for field_id, item in (source.get("fields") or {}).items()
        }
        boundary_centers = {
            (item.get("field_id"), item.get("boundary_id")): _point(item.get("center"))
            for item in (source.get("boundaries") or [])
        }
        return uav_xy, depot_xy, field_centers, boundary_centers

    uavs = getattr(source, "uavs", None) or []
    uav = uavs[0] if uavs else None
    uav_xy = _point(getattr(uav, "current_position", None))
    depot = getattr(source, "depot", None)
    depot_xy = None
    if depot is not None:
        getter = getattr(depot, "get_position", None)
        raw = getter() if callable(getter) else getattr(depot, "position", None)
        depot_xy = _point(raw)

    field_centers: dict = {}
    boundary_centers: dict = {}
    for field in getattr(source, "fields", None) or []:
        field_centers[field.field_id] = _point(field.centroid)
        for boundary in field.boundaries:
            boundary_centers[(field.field_id, boundary.id)] = _point(boundary.center)
    return uav_xy, depot_xy, field_centers, boundary_centers


def build_action_spatial_feature(actions, source, *, scales=None) -> torch.Tensor:
    """Return one row per action: ``[uav->field, uav->entry, depot->field,
    depot->entry, entry->exit]``, each normalized into ``[0, 1]``."""
    scale = float((scales or DEFAULT_SCALES).get("distance_scale_m", 2_000.0))
    if scale <= 0.0:
        raise ValueError("distance_scale_m must be positive.")
    uav_xy, depot_xy, field_centers, boundary_centers = _tables(source)
    rows = []
    for action in actions:
        field_xy = field_centers.get(action.field_id)
        entry_xy = boundary_centers.get((action.field_id, action.entry_boundary_id))
        exit_xy = boundary_centers.get((action.field_id, action.exit_boundary_id))
        rows.append([
            _norm(_distance(uav_xy, field_xy), scale),
            _norm(_distance(uav_xy, entry_xy), scale),
            _norm(_distance(depot_xy, field_xy), scale),
            _norm(_distance(depot_xy, entry_xy), scale),
            _norm(_distance(entry_xy, exit_xy), scale),
        ])
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, ACTION_SPATIAL_DIM)
