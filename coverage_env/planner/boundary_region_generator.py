"""Grow workload regions outward from an entry/exit-guided field edge."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import hypot, isfinite, sqrt

from shapely.geometry import LineString, Point

from coverage_env.geometry.discretization import GridPoint
from coverage_env.planner.growth_edge import GrowthEdgeSelection, GrowthEdgeSelector


@dataclass(frozen=True)
class BoundaryRegionSelection:
    """Connected grid region, growth order, and requested/available area."""

    region: list[GridPoint]
    scan_order: list[GridPoint]
    required_area: float
    selected_area: float
    max_reachable_area: float
    growth_edge: object
    growth_diagnostics: GrowthEdgeSelection


class BoundaryRegionGenerator:
    """Build a connected prefix of uncovered cells from a selected growth edge.

    Seeds are the connected edge-adjacent cells on the growth edge. A priority
    frontier adds only neighboring cells, ordered first by inward depth and
    then by offset along the edge. The selected region therefore remains
    connected and behaves like an expanding boundary strip, rather than a
    sorted set of arbitrary cells.
    """

    def __init__(self, growth_edge_selector: GrowthEdgeSelector | None = None) -> None:
        self.growth_edge_selector = growth_edge_selector or GrowthEdgeSelector()

    def generate(
        self,
        field,
        entry_edge,
        exit_edge,
        workload_ratio: float,
        *,
        workload_reference_area: float | None = None,
    ) -> BoundaryRegionSelection:
        """Select a connected coverage prefix or reject an oversized request."""
        if not isfinite(workload_ratio) or not 0.0 < workload_ratio <= 1.0:
            raise ValueError("workload_ratio must be finite and in (0, 1].")
        if not field.grid_points:
            raise ValueError("Field must be discretized before region generation.")

        diagnostics = self.growth_edge_selector.select(field, entry_edge, exit_edge)
        growth_edge = diagnostics.growth_edge
        expansion_order = self._expansion_order(field, growth_edge)
        if not expansion_order:
            raise ValueError("Growth edge has no connected uncovered grid cells.")

        max_area = sum(point.point_area for point in expansion_order)
        if workload_reference_area is None:
            workload_reference_area = sum(
                point.point_area for point in field.grid_points if not point.covered
            )
        reference_area = float(workload_reference_area)
        if not isfinite(reference_area) or reference_area <= 0.0:
            raise ValueError("workload_reference_area must be finite and positive.")
        required_area = workload_ratio * reference_area
        if required_area > max_area + 1e-8:
            raise ValueError(
                "Requested workload exceeds the maximum connected region "
                f"reachable from growth edge {growth_edge.id} "
                f"({required_area:.3f} > {max_area:.3f} m^2)."
            )

        region: list[GridPoint] = []
        selected_area = 0.0
        for point in expansion_order:
            region.append(point)
            selected_area += point.point_area
            if selected_area + 1e-8 >= required_area:
                break
        return BoundaryRegionSelection(
            region=region,
            scan_order=expansion_order,
            required_area=required_area,
            selected_area=selected_area,
            max_reachable_area=max_area,
            growth_edge=growth_edge,
            growth_diagnostics=diagnostics,
        )

    def max_reachable_area(self, field, growth_edge) -> float:
        """Return area in the single connected component seeded at an edge."""
        return float(sum(
            point.point_area for point in self._expansion_order(field, growth_edge)
        ))

    def _expansion_order(self, field, growth_edge) -> list[GridPoint]:
        """Return one connected flood-growth ordering from an edge seed band."""
        available = [point for point in field.grid_points if not point.covered]
        if not available:
            return []
        by_id = {point.id: point for point in available}
        resolution = float(field.grid_resolution or max(sqrt(p.point_area) for p in available))
        edge_line = LineString([growth_edge.start_point, growth_edge.end_point])
        distances = {
            point.id: edge_line.distance(Point(point.x, point.y))
            for point in available
        }
        minimum_distance = min(distances.values())
        seed_limit = minimum_distance + max(resolution * 0.8, 1e-6)
        seed_candidates = {
            point.id for point in available if distances[point.id] <= seed_limit
        }
        if not seed_candidates:
            seed_candidates = {min(available, key=lambda p: distances[p.id]).id}

        # If a diagonal or very short edge makes multiple seed islands, keep
        # the largest connected seed island; this prevents disconnected starts.
        seed_components = []
        remaining_seeds = set(seed_candidates)
        while remaining_seeds:
            root = min(remaining_seeds)
            component = self._component(root, seed_candidates, by_id)
            seed_components.append(component)
            remaining_seeds.difference_update(component)
        seeds = max(seed_components, key=lambda group: (len(group), sorted(group)[0]))

        inward = (-growth_edge.normal_vector[0], -growth_edge.normal_vector[1])
        tangent = growth_edge.tangent_vector
        origin = growth_edge.center

        def priority(point: GridPoint) -> tuple[float, float, str]:
            dx, dy = point.x - origin[0], point.y - origin[1]
            depth = max(0.0, dx * inward[0] + dy * inward[1])
            along = abs(dx * tangent[0] + dy * tangent[1])
            return depth, along, point.id

        # A seed component can be connected while an arbitrary priority sort
        # of all its cells is not prefix-connected. Traverse the seed
        # component from one root so every workload prefix remains connected.
        selected_ids: set[str] = set()
        order: list[GridPoint] = []
        seed_queue = []
        seed_serial = count()
        seed_root = min((by_id[point_id] for point_id in seeds), key=priority)
        heappush(seed_queue, (*priority(seed_root), next(seed_serial)))
        while seed_queue:
            _depth, _along, point_id, _serial = heappop(seed_queue)
            if point_id in selected_ids:
                continue
            selected_ids.add(point_id)
            point = by_id[point_id]
            order.append(point)
            for neighbor_id in point.neighbors:
                if neighbor_id in seeds and neighbor_id not in selected_ids:
                    neighbor = by_id[neighbor_id]
                    heappush(seed_queue, (*priority(neighbor), next(seed_serial)))

        queue = []
        serial = count()

        def enqueue_neighbors(point: GridPoint) -> None:
            for neighbor_id in point.neighbors:
                if neighbor_id in by_id and neighbor_id not in selected_ids:
                    candidate = by_id[neighbor_id]
                    depth, along, point_id = priority(candidate)
                    heappush(queue, (depth, along, point_id, next(serial)))

        for seed in order:
            enqueue_neighbors(seed)
        while queue:
            _depth, _along, point_id, _serial = heappop(queue)
            if point_id in selected_ids or point_id not in by_id:
                continue
            selected_ids.add(point_id)
            point = by_id[point_id]
            order.append(point)
            enqueue_neighbors(point)
        return order

    @staticmethod
    def _component(root, allowed_ids, by_id):
        reached = {root}
        pending = [root]
        while pending:
            point_id = pending.pop()
            for neighbor_id in by_id[point_id].neighbors:
                if neighbor_id in allowed_ids and neighbor_id not in reached:
                    reached.add(neighbor_id)
                    pending.append(neighbor_id)
        return reached

