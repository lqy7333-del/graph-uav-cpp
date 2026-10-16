"""Construction of Field-Boundary and Boundary-Boundary relations."""

from __future__ import annotations

from math import atan2, cos, hypot, radians, sin

import torch


EDGE_FB = 0
EDGE_BB_INTRA = 1
EDGE_BB_INTER = 2
# The UAV as a graph node. Direction is UAV -> Field / UAV -> Boundary, so each
# candidate destination receives "where is the agent relative to me" as a
# message whose *content and attention* are both functions of the relative
# offset and the normalized distance.
EDGE_UF = 3
EDGE_UB = 4

RELATION_FEATURE_DIMS = {
    EDGE_FB: 4,
    EDGE_BB_INTRA: 3,
    EDGE_BB_INTER: 3,
    EDGE_UF: 3,
    EDGE_UB: 3,
}


def _axis_angle(tangent) -> float:
    """Return a segment's unoriented angle in radians, in [0, pi)."""
    return atan2(tangent[1], tangent[0]) % 3.141592653589793


def _axis_difference(first: float, second: float) -> float:
    """Return the smaller angle between unoriented axes, normalized by pi/2."""
    raw = abs(first - second) % 3.141592653589793
    return min(raw, 3.141592653589793 - raw) / (3.141592653589793 / 2.0)


class RelationFeatureBuilder:
    """Build exactly the three permitted relation families.

    ``edge_index`` uses a unified node index space: Field nodes first, followed
    by Boundary nodes. Each relation's raw features remain separately
    available because their requested dimensions differ.
    """

    def __init__(
        self,
        d_near: float = 1_500.0,
        global_distance_scale: float | None = None,
        uav_distance_scale: float = 3_000.0,
    ):
        if d_near <= 0.0:
            raise ValueError("d_near must be positive.")
        if global_distance_scale is not None and global_distance_scale <= 0.0:
            raise ValueError("global_distance_scale must be positive when supplied.")
        if uav_distance_scale <= 0.0:
            raise ValueError("uav_distance_scale must be positive.")
        self.d_near = float(d_near)
        self.global_distance_scale = global_distance_scale
        self.uav_distance_scale = float(uav_distance_scale)

    def build(self, environment, field_id_to_index, boundary_id_to_index):
        """Return edge arrays, typed relation features, and relation counts."""
        fields = list(environment.fields)
        nf = len(fields)
        boundary_records = [
            (field, boundary)
            for field in fields
            for boundary in field.boundaries
        ]
        global_scale = self.global_distance_scale or self._environment_scale(fields)
        edges: list[tuple[int, int]] = []
        edge_types: list[int] = []
        padded_attrs: list[list[float]] = []
        features = {
            EDGE_FB: [], EDGE_BB_INTRA: [], EDGE_BB_INTER: [], EDGE_UF: [], EDGE_UB: [],
        }
        counts = {key: 0 for key in features}

        def add_edge(source, target, relation_type, values):
            edges.append((source, target))
            edge_types.append(relation_type)
            features[relation_type].append(values)
            padded_attrs.append(values + [0.0] * (4 - len(values)))
            counts[relation_type] += 1

        # E_FB: one directed Field -> owned Boundary edge.
        for field, boundary in boundary_records:
            field_idx = field_id_to_index[field.field_id]
            boundary_idx = nf + boundary_id_to_index[field.field_id][boundary.id]
            field_scale = max(hypot(
                field.bounding_box[2] - field.bounding_box[0],
                field.bounding_box[3] - field.bounding_box[1],
            ), 1e-9)
            relative_x = (boundary.center[0] - field.centroid[0]) / field_scale
            relative_y = (boundary.center[1] - field.centroid[1]) / field_scale
            b_angle = _axis_angle(boundary.tangent_vector)
            f_angle = radians(field.orientation)
            angle_difference = _axis_difference(b_angle, f_angle % 3.141592653589793)
            add_edge(field_idx, boundary_idx, EDGE_FB, [
                boundary.length / max(field.perimeter, 1e-9),
                relative_x, relative_y, angle_difference,
            ])

        # E_BB_intra: all ordered, non-self Boundary pairs in each Field.
        for field in fields:
            boundaries = field.boundaries
            field_scale = max(hypot(
                field.bounding_box[2] - field.bounding_box[0],
                field.bounding_box[3] - field.bounding_box[1],
            ), 1e-9)
            field_angle = radians(field.orientation) % 3.141592653589793
            for first in boundaries:
                for second in boundaries:
                    if first.id == second.id:
                        continue
                    first_idx = nf + boundary_id_to_index[field.field_id][first.id]
                    second_idx = nf + boundary_id_to_index[field.field_id][second.id]
                    distance = hypot(
                        first.center[0] - second.center[0],
                        first.center[1] - second.center[1],
                    )
                    first_angle = _axis_angle(first.tangent_vector)
                    second_angle = _axis_angle(second.tangent_vector)
                    angle_difference = _axis_difference(first_angle, second_angle)
                    alignment_first = abs(cos(first_angle - field_angle))
                    alignment_second = abs(cos(second_angle - field_angle))
                    direction_similarity = (alignment_first + alignment_second) / 2.0
                    add_edge(first_idx, second_idx, EDGE_BB_INTRA, [
                        distance / field_scale, angle_difference, direction_similarity,
                    ])

        # E_BB_inter: directed cross-field pairs within D_near only.
        for first_field, first in boundary_records:
            for second_field, second in boundary_records:
                if first_field.field_id == second_field.field_id:
                    continue
                distance = hypot(
                    first.center[0] - second.center[0],
                    first.center[1] - second.center[1],
                )
                if distance >= self.d_near:
                    continue
                first_idx = nf + boundary_id_to_index[first_field.field_id][first.id]
                second_idx = nf + boundary_id_to_index[second_field.field_id][second.id]
                angle_difference = _axis_difference(
                    _axis_angle(first.tangent_vector),
                    _axis_angle(second.tangent_vector),
                )
                normal_similarity = (
                    first.normal_vector[0] * second.normal_vector[0]
                    + first.normal_vector[1] * second.normal_vector[1]
                )
                add_edge(first_idx, second_idx, EDGE_BB_INTER, [
                    distance / global_scale, angle_difference, normal_similarity,
                ])

        # E_UF / E_UB: the UAV node points at every Field and every Boundary.
        # This is what lets "which candidate is closest" be computed *by the
        # graph* -- the distance is an edge feature, so it enters both the
        # attention score and the message content of the destination node.
        uav_position = self._uav_position(environment, fields)
        if uav_position is not None:
            uav_index = nf + len(boundary_records)
            scale = self.uav_distance_scale
            for field in fields:
                offset_x = field.centroid[0] - uav_position[0]
                offset_y = field.centroid[1] - uav_position[1]
                add_edge(uav_index, field_id_to_index[field.field_id], EDGE_UF, [
                    offset_x / scale, offset_y / scale, hypot(offset_x, offset_y) / scale,
                ])
            for field, boundary in boundary_records:
                offset_x = boundary.center[0] - uav_position[0]
                offset_y = boundary.center[1] - uav_position[1]
                boundary_idx = nf + boundary_id_to_index[field.field_id][boundary.id]
                add_edge(uav_index, boundary_idx, EDGE_UB, [
                    offset_x / scale, offset_y / scale, hypot(offset_x, offset_y) / scale,
                ])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return {
            "edge_index": edge_index,
            "edge_type": torch.tensor(edge_types, dtype=torch.long),
            "edge_attr": torch.tensor(padded_attrs, dtype=torch.float32).reshape(-1, 4),
            "relation_features": {
                key: torch.tensor(value, dtype=torch.float32).reshape(
                    -1, RELATION_FEATURE_DIMS[key]
                )
                for key, value in features.items()
            },
            "relation_counts": counts,
            "relation_feature_dims": dict(RELATION_FEATURE_DIMS),
        }

    @staticmethod
    def _uav_position(environment, fields):
        """Return the UAV's current position, falling back to the depot."""
        uavs = getattr(environment, "uavs", None) or []
        if uavs:
            position = getattr(uavs[0], "current_position", None)
            if position is not None:
                return (float(position[0]), float(position[1]))
        depot = getattr(environment, "depot", None)
        if depot is not None:
            getter = getattr(depot, "get_position", None)
            raw = getter() if callable(getter) else None
            if raw is not None:
                return (float(raw[0]), float(raw[1]))
        if fields:
            return (float(fields[0].centroid[0]), float(fields[0].centroid[1]))
        return None

    @staticmethod
    def _environment_scale(fields) -> float:
        """Return a stable global distance scale from field center spread."""
        if not fields:
            return 1.0
        xs = [field.centroid[0] for field in fields]
        ys = [field.centroid[1] for field in fields]
        return max(hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
