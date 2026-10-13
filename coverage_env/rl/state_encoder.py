"""Stable graph-shaped observation encoding for future GNN integration."""

from __future__ import annotations


class StateEncoder:
    """Convert Environment dictionaries into ordered node/edge feature lists.

    This is a data adapter only. It deliberately has no tensor or GNN
    dependency, so a future policy can choose its own framework.
    """

    @staticmethod
    def encode(state: dict) -> dict:
        """Return UAV, field, boundary features and field-boundary edges."""
        uavs = list(state.get("uavs", {}).values())
        fields = list(state.get("fields", {}).values())
        boundaries = list(state.get("boundaries", []))
        field_indices = {field_id: index for index, field_id in enumerate(
            item["field_id"] for item in fields
        )}
        return {
            "uav_ids": list(state.get("uavs", {}).keys()),
            "uav_features": [
                [float((item.get("position") or (0.0, 0.0))[0]),
                 float((item.get("position") or (0.0, 0.0))[1]),
                 float(item.get("remaining_energy", 0.0)),
                 float(item.get("total_distance", 0.0))]
                for item in uavs
            ],
            "field_ids": [item["field_id"] for item in fields],
            "field_features": [
                [float(item.get("remaining_area", 0.0)),
                 float(item.get("coverage_ratio", 0.0)),
                 float(item.get("center", (0.0, 0.0))[0]),
                 float(item.get("center", (0.0, 0.0))[1]),
                 float(item.get("orientation", 0.0))]
                for item in fields
            ],
            "boundary_ids": [
                (item["field_id"], item["boundary_id"]) for item in boundaries
            ],
            "boundary_features": [
                [float(item.get("length", 0.0)), float(item.get("available", False))]
                for item in boundaries
            ],
            "field_boundary_edges": [
                (field_indices[item["field_id"]], index)
                for index, item in enumerate(boundaries)
            ],
            "depot": state.get("depot", {}).get("position"),
        }
