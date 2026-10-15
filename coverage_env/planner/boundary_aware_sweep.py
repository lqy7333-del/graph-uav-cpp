"""Boundary-constrained sweep direction and entry/exit path optimization."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, hypot, radians, sin

from shapely.affinity import rotate
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from coverage_env.planner.coverage_path import _from_scan_coordinates


@dataclass(frozen=True)
class BoundaryAwareSweepPlan:
    """Selected scan path and its optimized boundary endpoints."""

    orientation: float
    coverage_path: list[tuple[float, float]]
    coverage_strokes: list[tuple[tuple[float, float], tuple[float, float]]]
    scan_transit_paths: list[list[tuple[float, float]]]
    entry_point: tuple[float, float]
    exit_point: tuple[float, float]
    total_cost: float
    region_geometry: object


class BoundaryAwareSweepPlanner:
    """Jointly select a finite sweep direction and scan-line orientation.

    For each candidate axis, the clipped coverage geometry is decomposed into
    parallel scan intervals. A two-state dynamic program chooses each interval
    direction, charging the entry-boundary connection, inter-stroke distance,
    and terminal exit-boundary connection. The first and last points are
    therefore optimized against the configured edges instead of projecting an
    arbitrary path after planning.
    """

    def plan(
        self,
        region,
        field,
        entry_edge,
        exit_edge,
        *,
        current_position,
        lane_spacing: float,
        candidate_directions=None,
    ) -> BoundaryAwareSweepPlan:
        """Return a coverage route whose endpoints satisfy both edge limits."""
        if not region:
            raise ValueError("Cannot plan a sweep over an empty coverage region.")
        if lane_spacing <= 0.0:
            raise ValueError("lane_spacing must be positive.")
        region_geometry = unary_union([
            box(
                point.x - point.point_area ** 0.5 / 2.0,
                point.y - point.point_area ** 0.5 / 2.0,
                point.x + point.point_area ** 0.5 / 2.0,
                point.y + point.point_area ** 0.5 / 2.0,
            )
            for point in region
        ]).buffer(1e-6).intersection(field.xy_polygon)
        if region_geometry.is_empty:
            raise ValueError("Coverage points produced an empty sweep geometry.")

        best = None
        directions = (
            [float(value) % 180.0 for value in candidate_directions]
            if candidate_directions is not None
            else self._candidate_directions(field, entry_edge, exit_edge)
        )
        # Every candidate in this call shares `region_geometry`, so the
        # polygon-only visibility pairs are computed once and reused by all of
        # its connectors. The dict lives no longer than this geometry.
        visibility_pairs: dict = {}
        for orientation in directions:
            lane_data = self._scan_intervals(
                region_geometry, orientation, lane_spacing, region
            )
            if not lane_data:
                continue
            for reverse_lanes in (False, True):
                ordered = list(reversed(lane_data)) if reverse_lanes else list(lane_data)
                strokes = []
                for _lane, segments in ordered:
                    strokes.extend(
                        reversed(segments) if reverse_lanes else segments
                    )
                if not strokes:
                    continue
                try:
                    ordered_strokes, lower_bound = self._choose_stroke_order(
                        strokes, orientation, entry_edge, exit_edge,
                        current_position,
                    )
                    # A straight-line DP score is a lower bound on the route
                    # cost. Avoid expensive visibility-graph connectors for
                    # candidates that cannot beat the best feasible route.
                    if best is not None and lower_bound >= best.total_cost - 1e-9:
                        continue
                    candidate = self._materialize_strokes(
                        ordered_strokes, orientation, entry_edge, exit_edge,
                        current_position, region_geometry, lower_bound,
                        visibility_pairs,
                    )
                except ValueError:
                    # Some scan axes can divide a concave sampled region into
                    # pieces that cannot be joined without leaving the region.
                    # Keep searching the finite candidate set.
                    continue
                if best is None or candidate.total_cost < best.total_cost:
                    best = candidate
        if best is None:
            raise ValueError("No non-empty scanline intersects the coverage region.")
        return best

    @staticmethod
    def _candidate_directions(field, entry_edge, exit_edge):
        """Field axes, entry-exit axis, and all field edge axes, modulo 180°."""
        values = [field.orientation, field.orientation + 90.0]
        dx = exit_edge.center[0] - entry_edge.center[0]
        dy = exit_edge.center[1] - entry_edge.center[1]
        if hypot(dx, dy) > 1e-9:
            values.append(atan2(dy, dx) * 180.0 / 3.141592653589793)
        values.extend(
            atan2(edge.tangent_vector[1], edge.tangent_vector[0])
            * 180.0 / 3.141592653589793
            for edge in field.boundaries if edge.ring_id == 0
        )
        unique = []
        for angle in values:
            axis = float(angle) % 180.0
            if not any(min(abs(axis - old), 180.0 - abs(axis - old)) < 1e-7
                       for old in unique):
                unique.append(axis)
        return unique

    @staticmethod
    def _scan_intervals(region_geometry, orientation, lane_spacing, region=()):
        """Intersect parallel scan lines with rotated coverage geometry."""
        scan_geometry = rotate(region_geometry, -orientation, origin=(0.0, 0.0))
        min_u, min_v, max_u, max_v = scan_geometry.bounds
        width = max_v - min_v
        lane_count = max(1, int(ceil(width / lane_spacing)))
        # A clipped edge fragment may be much narrower than the UAV swath.
        # Put its sole scanline at the fragment midpoint instead of offsetting
        # it by half a full swath, which would miss the region entirely.
        spacing = width / lane_count if width > 1e-9 else 0.0
        half_length = max(max_u - min_u, width, lane_spacing, 1.0) + lane_spacing
        lane_values = [
            min_v + (lane_index + 0.5) * spacing
            for lane_index in range(lane_count)
        ]
        # The regular lane centers can all fall in gaps of a narrow,
        # fragmented, or concave region. Add a line through any coverage-cell
        # center that would otherwise sit more than half a swath from every
        # regular lane. This keeps the requested cells under an actual sweep.
        theta = radians(orientation)
        normal = (-sin(theta), cos(theta))
        for point in region:
            point_v = point.x * normal[0] + point.y * normal[1]
            if min(abs(point_v - lane_v) for lane_v in lane_values) > lane_spacing / 2.0 + 1e-9:
                lane_values.append(point_v)
        lane_values.sort()

        lanes = []
        for lane_v in lane_values:
            line = LineString([
                (min_u - half_length, lane_v), (max_u + half_length, lane_v)
            ])
            intersection = scan_geometry.intersection(line)
            segments = []
            for part in BoundaryAwareSweepPlanner._line_parts(intersection):
                coords = list(part.coords)
                if len(coords) < 2:
                    continue
                low, high = min(p[0] for p in coords), max(p[0] for p in coords)
                if high - low > 1e-8:
                    segments.append((
                        _from_scan_coordinates(low, lane_v, orientation),
                        _from_scan_coordinates(high, lane_v, orientation),
                    ))
            segments.sort(key=lambda item: item[0][0] * cos(radians(orientation))
                          + item[0][1] * sin(radians(orientation)))
            if segments:
                lanes.append((lane_v, segments))
        return lanes

    @staticmethod
    def _line_parts(geometry):
        if geometry.is_empty:
            return
        if geometry.geom_type == "LineString":
            yield geometry
        elif hasattr(geometry, "geoms"):
            for part in geometry.geoms:
                yield from BoundaryAwareSweepPlanner._line_parts(part)

    def _choose_stroke_order(
        self, strokes, orientation, entry_edge, exit_edge,
        current_position,
    ):
        """Choose stroke directions using a two-state DP and straight costs."""
        # State 0 traverses segment[0] -> segment[1], state 1 reverses it.
        n = len(strokes)
        costs = [[float("inf"), float("inf")] for _ in range(n)]
        parents = [[None, None] for _ in range(n)]

        def endpoints(index, state):
            first, last = strokes[index]
            return (first, last) if state == 0 else (last, first)

        for state in (0, 1):
            start, end = endpoints(0, state)
            boundary_start = self._project_to_segment(start, entry_edge)
            costs[0][state] = (
                self._distance(current_position, boundary_start)
                + self._distance(boundary_start, start)
                + self._distance(start, end)
            )

        for index in range(1, n):
            for state in (0, 1):
                start, end = endpoints(index, state)
                for previous_state in (0, 1):
                    _previous_start, previous_end = endpoints(index - 1, previous_state)
                    candidate = costs[index - 1][previous_state] + self._distance(
                        previous_end, start
                    ) + self._distance(start, end)
                    if candidate < costs[index][state]:
                        costs[index][state] = candidate
                        parents[index][state] = previous_state

        best_state, best_cost = None, float("inf")
        for state in (0, 1):
            _start, end = endpoints(n - 1, state)
            boundary_end = self._project_to_segment(end, exit_edge)
            candidate = costs[n - 1][state] + self._distance(end, boundary_end)
            if candidate < best_cost:
                best_state, best_cost = state, candidate

        states = [0] * n
        states[-1] = best_state
        for index in range(n - 1, 0, -1):
            states[index - 1] = parents[index][states[index]]

        ordered_strokes = [endpoints(index, states[index]) for index in range(n)]
        return ordered_strokes, float(best_cost)

    def _materialize_strokes(
        self, ordered_strokes, orientation, entry_edge, exit_edge,
        current_position, region_geometry, lower_bound, visibility_pairs=None,
    ):
        """Build actual in-region connectors for one promising DP candidate."""
        route = [ordered_strokes[0][0]]
        connectors = []
        for start, end in ordered_strokes:
            if route[-1] != start:
                connector = self._connector(
                    route[-1], start, region_geometry, visibility_pairs
                )
                connectors.append(connector)
                route.extend(connector[1:])
            if route[-1] != end:
                route.append(end)
        entry_point = self._project_to_segment(route[0], entry_edge)
        exit_point = self._project_to_segment(route[-1], exit_edge)
        actual_cost = (
            self._distance(current_position, entry_point)
            + self._distance(entry_point, route[0])
            + sum(self._distance(start, end) for start, end in zip(route[:-1], route[1:]))
            + self._distance(route[-1], exit_point)
        )
        return BoundaryAwareSweepPlan(
            orientation=float(orientation),
            coverage_path=self._dedupe(route),
            coverage_strokes=ordered_strokes,
            scan_transit_paths=connectors,
            entry_point=entry_point,
            exit_point=exit_point,
            total_cost=float(max(lower_bound, actual_cost)),
            region_geometry=region_geometry,
        )

    def _connector(self, start, end, geometry, visibility_pairs=None):
        """Connect strokes inside their generated coverage geometry."""
        if start == end:
            return [start]
        direct = LineString([start, end])
        if geometry.buffer(1e-6).covers(direct):
            return [start, end]
        # The legacy visibility-graph routine handles polygonal components;
        # when it cannot connect the points, report infeasibility instead of
        # silently drawing a chord through uncovered geometry.
        from coverage_env.planner.coverage_path import _shortest_path_inside_polygon

        path = _shortest_path_inside_polygon(
            start, end, geometry, pair_cache=visibility_pairs
        )
        if not geometry.buffer(1e-5).covers(LineString(path)):
            raise ValueError("No connector exists inside the generated coverage region.")
        return path

    @staticmethod
    def _project_to_segment(point, edge):
        segment = LineString([edge.start_point, edge.end_point])
        projected = segment.interpolate(segment.project(Point(point)))
        return float(projected.x), float(projected.y)

    @staticmethod
    def _distance(first, second):
        return hypot(first[0] - second[0], first[1] - second[1])

    @staticmethod
    def _dedupe(points):
        result = []
        for point in points:
            if not result or point != result[-1]:
                result.append(point)
        return result

