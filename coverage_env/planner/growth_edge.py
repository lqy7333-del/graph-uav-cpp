"""Select a boundary edge that can grow a connected coverage strip."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class GrowthEdgeSelection:
    """Chosen growth edge together with the finite candidate diagnostics."""

    growth_edge: object
    candidate_ids: tuple[str, ...]
    corridor_depths: dict[str, float]
    used_convex_geometry: bool


class GrowthEdgeSelector:
    """Choose a non-entry/non-exit edge on the shorter boundary chain.

    For convex fields, ``corridor_depth`` is the greatest inward distance from
    a candidate edge to any polygon vertex. The maximum-depth candidate gives
    a wide strip to grow from. For non-convex fields the same finite heuristic
    is retained as a fallback; the region generator still enforces grid
    connectivity.
    """

    def select(self, field, entry_edge, exit_edge) -> GrowthEdgeSelection:
        """Return the growth edge implied by entry/exit edge constraints."""
        exterior_edges = [edge for edge in field.boundaries if edge.ring_id == 0]
        if not exterior_edges:
            raise ValueError("Field has no exterior boundary edges.")
        by_id = {edge.id: index for index, edge in enumerate(exterior_edges)}
        if entry_edge.id not in by_id or exit_edge.id not in by_id:
            raise ValueError("Entry and exit edges must belong to the exterior ring.")

        entry_index, exit_index = by_id[entry_edge.id], by_id[exit_edge.id]
        count = len(exterior_edges)
        forward = self._chain(entry_index, exit_index, 1, count)
        backward = self._chain(entry_index, exit_index, -1, count)
        if forward and backward:
            if len(forward) < len(backward):
                candidate_indices = forward
            elif len(backward) < len(forward):
                candidate_indices = backward
            else:
                candidate_indices = sorted(set(forward + backward))
        else:
            # Adjacent entry/exit edges leave one empty chain. Use the other
            # side so the growth edge remains distinct from both constraints.
            candidate_indices = forward or backward

        if not candidate_indices:
            candidate_indices = [
                index for index in range(count)
                if index not in {entry_index, exit_index}
            ]
        if not candidate_indices:
            raise ValueError("No boundary edge is available as a growth edge.")

        candidates = [exterior_edges[index] for index in candidate_indices]
        depths = {edge.id: self.corridor_depth(field, edge) for edge in candidates}
        convex = self._is_convex(field)
        chosen = max(
            candidates,
            key=lambda edge: (depths[edge.id], edge.length, -by_id[edge.id]),
        )
        return GrowthEdgeSelection(
            growth_edge=chosen,
            candidate_ids=tuple(edge.id for edge in candidates),
            corridor_depths=depths,
            used_convex_geometry=convex,
        )

    @staticmethod
    def _chain(start: int, stop: int, step: int, count: int) -> list[int]:
        """Indices strictly between two edges while walking around a ring."""
        indices = []
        index = (start + step) % count
        while index != stop:
            indices.append(index)
            index = (index + step) % count
        return indices

    @staticmethod
    def _is_convex(field) -> bool:
        polygon = field.xy_polygon
        if polygon.interiors:
            return False
        hull = polygon.convex_hull
        return abs(hull.area - polygon.area) <= max(1e-7, polygon.area * 1e-9)

    @staticmethod
    def corridor_depth(field, edge) -> float:
        """Return the maximum inward projection of the field vertices."""
        inward = (-edge.normal_vector[0], -edge.normal_vector[1])
        origin = edge.center
        coordinates = list(field.xy_polygon.exterior.coords)[:-1]
        if not coordinates:
            return 0.0
        return max(
            0.0,
            max(
                (x - origin[0]) * inward[0] + (y - origin[1]) * inward[1]
                for x, y in coordinates
            ),
        )

