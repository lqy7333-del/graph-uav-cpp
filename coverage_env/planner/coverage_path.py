"""Boustrophedon coverage paths from parallel clipped scan lines."""

from __future__ import annotations

from heapq import heappop, heappush
from math import ceil, cos, hypot, radians, sin, sqrt
from typing import Sequence

from shapely.affinity import rotate
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from coverage_env.geometry.discretization import GridPoint

Point2D = tuple[float, float]


class SweepPlan:
    """Coverage strokes and the separate feasible connectors between them."""

    def __init__(self, path, strokes, connectors):
        self.path = path
        self.strokes = strokes
        self.connectors = connectors


def order_grid_points(
    points: Sequence[GridPoint],
    orientation: float,
    *,
    start_point: Point2D | None = None,
    target_point: Point2D | None = None,
) -> list[GridPoint]:
    """Return points in approximate alternating scan-line order."""
    if not points:
        return []
    theta = radians(orientation)
    axis = (cos(theta), sin(theta))
    normal = (-axis[1], axis[0])
    resolution = max(sqrt(point.point_area) for point in points)
    band_width = max(resolution * 0.5, 1e-6)
    projected = {
        point.id: (
            point.x * axis[0] + point.y * axis[1],
            point.x * normal[0] + point.y * normal[1],
        )
        for point in points
    }
    bands: dict[int, list[GridPoint]] = {}
    for point in points:
        _, cross_track = projected[point.id]
        band = round(cross_track / band_width)
        bands.setdefault(band, []).append(point)

    start_cross_track = (
        start_point[0] * normal[0] + start_point[1] * normal[1]
        if start_point is not None else None
    )
    target_cross_track = (
        target_point[0] * normal[0] + target_point[1] * normal[1]
        if target_point is not None else None
    )
    band_values = {
        band: sum(projected[point.id][1] for point in group) / len(group)
        for band, group in bands.items()
    }
    if start_cross_track is None:
        first_band = min(bands, key=lambda band: band_values[band])
    else:
        first_band = min(bands, key=lambda band: abs(band_values[band] - start_cross_track))

    if target_cross_track is not None and target_cross_track != start_cross_track:
        direction = 1 if target_cross_track > start_cross_track else -1
    elif start_cross_track is not None:
        direction = 1 if band_values[first_band] <= sum(band_values.values()) / len(band_values) else -1
    else:
        direction = 1

    def band_order_key(band: int) -> tuple[int, float, int]:
        delta = band_values[band] - band_values[first_band]
        toward_target = delta * direction >= -1e-9
        return (0 if toward_target else 1, abs(delta), band)

    ordered: list[GridPoint] = []
    previous_direction: int | None = None
    for band in sorted(bands, key=band_order_key):
        group = bands[band]
        group.sort(key=lambda point: projected[point.id][0])
        if not ordered and start_point is not None:
            first_u = projected[group[0].id][0]
            last_u = projected[group[-1].id][0]
            start_u = start_point[0] * axis[0] + start_point[1] * axis[1]
            if abs(last_u - start_u) < abs(first_u - start_u):
                group.reverse()
                previous_direction = -1
            else:
                previous_direction = 1
        elif previous_direction == 1:
            group.reverse()
            previous_direction = -1
        else:
            previous_direction = 1
        ordered.extend(group)
    return ordered


def generate_boustrophedon_path(
    coverage_region: Sequence[GridPoint],
    orientation: float,
    *,
    start_point: Point2D | None = None,
    lane_spacing: float | None = None,
    field_polygon: BaseGeometry | None = None,
    target_boundary=None,
    coverage_start_point: Point2D | None = None,
    entry_boundary=None,
) -> list[Point2D]:
    """Sweep a connected point region with alternating parallel scan lines.

    Each grid point represents a square cell. Their union is clipped to the
    field polygon when supplied, rotated into scan coordinates, and intersected
    with parallel lines. Consecutive lanes reverse direction to form a true
    boustrophedon pattern. ``lane_spacing`` should match the UAV's effective
    swath width; it defaults to the point-grid resolution.
    """
    if coverage_start_point is not None:
        start_point = coverage_start_point
    return plan_boustrophedon_path(
        coverage_region,
        orientation,
        start_point=start_point,
        lane_spacing=lane_spacing,
        field_polygon=field_polygon,
        target_boundary=target_boundary,
        entry_boundary=entry_boundary,
    ).path


def plan_boustrophedon_path(
    coverage_region: Sequence[GridPoint],
    orientation: float,
    *,
    start_point: Point2D | None = None,
    lane_spacing: float | None = None,
    field_polygon: BaseGeometry | None = None,
    target_boundary=None,
    entry_boundary=None,
) -> SweepPlan:
    """Decompose scanlines into strokes and connect them with feasible paths."""
    if not coverage_region:
        return SweepPlan([], [], [])
    if lane_spacing is None:
        lane_spacing = max(sqrt(point.point_area) for point in coverage_region)
    if lane_spacing <= 0.0:
        raise ValueError("lane_spacing must be greater than zero.")

    cells = []
    for point in coverage_region:
        half_side = sqrt(point.point_area) / 2.0
        cells.append(
            box(point.x - half_side, point.y - half_side,
                point.x + half_side, point.y + half_side)
        )
    # Grid-cell edges can differ by a few floating-point ulps after projection.
    # A tiny metric buffer joins those seams before scanline intersection.
    region_geometry = unary_union(cells).buffer(1e-6)
    if field_polygon is not None:
        region_geometry = region_geometry.intersection(field_polygon)
    if region_geometry.is_empty:
        return SweepPlan([], [], [])

    scan_geometry = rotate(region_geometry, -orientation, origin=(0.0, 0.0))
    min_u, min_v, max_u, max_v = scan_geometry.bounds
    cross_track_width = max_v - min_v
    lane_count = max(1, ceil(cross_track_width / lane_spacing))
    effective_spacing = cross_track_width / lane_count if cross_track_width else lane_spacing
    lane_values = [
        min_v + (index + 0.5) * effective_spacing
        for index in range(lane_count)
    ]

    start_u = start_v = None
    if start_point is not None:
        theta = radians(orientation)
        start_u = start_point[0] * cos(theta) + start_point[1] * sin(theta)
        start_v = -start_point[0] * sin(theta) + start_point[1] * cos(theta)
        # Put the first scan line on the entry edge's cross-track coordinate.
        # When the selected region reaches that edge, the first sweep endpoint
        # is itself on the boundary and doubles as the coverage anchor.
        if min_v - 1e-7 <= start_v <= max_v + 1e-7:
            nearest_lane = min(
                range(len(lane_values)),
                key=lambda index: abs(lane_values[index] - start_v),
            )
            lane_values[nearest_lane] = start_v
        direction = 1 if abs(max_v - start_v) >= abs(start_v - min_v) else -1
        if target_boundary is not None:
            target_v = (
                -target_boundary.center[0] * sin(radians(orientation))
                + target_boundary.center[1] * cos(radians(orientation))
            )
            direction = 1 if target_v >= start_v else -1
        lane_values = [lane_values[0]] + sorted(
            lane_values[1:], key=lambda value: (value - start_v) * direction < 0
        )

    scanline_half_length = max(max_u - min_u, max_v - min_v, 1.0) + lane_spacing
    lanes_with_segments: list[tuple[float, list[tuple[float, float]]]] = []
    for lane_v in lane_values:
        scanline = LineString(
            [(min_u - scanline_half_length, lane_v),
             (max_u + scanline_half_length, lane_v)]
        )
        intersection = scan_geometry.intersection(scanline)
        segments: list[tuple[float, float]] = []
        for line in _line_parts(intersection):
            coordinates = list(line.coords)
            if len(coordinates) >= 2:
                low_u = min(coordinate[0] for coordinate in coordinates)
                high_u = max(coordinate[0] for coordinate in coordinates)
                if high_u - low_u > 1e-9:
                    segments.append((low_u, high_u))
        segments.sort()
        if segments:
            lanes_with_segments.append((lane_v, segments))

    if not lanes_with_segments:
        return SweepPlan([], [], [])

    entry_line = (
        LineString([entry_boundary.start_point, entry_boundary.end_point])
        if entry_boundary is not None else None
    )
    exit_line = (
        LineString([target_boundary.start_point, target_boundary.end_point])
        if target_boundary is not None else None
    )
    route: list[Point2D] = []
    ordered_strokes: list[tuple[Point2D, Point2D]] = []
    connectors: list[list[Point2D]] = []
    current = start_point
    for lane_index, (lane_v, lane_segments) in enumerate(lanes_with_segments):
        remaining = [
            (_from_scan_coordinates(low, lane_v, orientation),
             _from_scan_coordinates(high, lane_v, orientation))
            for low, high in lane_segments
        ]
        while remaining:
            choices = []
            for index, (first, last) in enumerate(remaining):
                for reverse, endpoint in ((False, first), (True, last)):
                    travel = hypot(endpoint[0] - current[0], endpoint[1] - current[1]) if current else 0.0
                    entry_penalty = (
                        entry_line.distance(Point(endpoint)) * 1e6
                        if not ordered_strokes and entry_line is not None else 0.0
                    )
                    exit_cost = 0.0
                    if lane_index == len(lanes_with_segments) - 1 and len(remaining) == 1 and exit_line is not None:
                        final_point = first if reverse else last
                        # Prefer finishing at the exit even when reversing the
                        # final stroke trades an equal connector distance.
                        exit_cost = 2.0 * exit_line.distance(Point(final_point))
                    choices.append((travel + entry_penalty + exit_cost, index, reverse))
            _, index, reverse = min(choices)
            first, last = remaining.pop(index)
            stroke = (last, first) if reverse else (first, last)
            if route and route[-1] != stroke[0]:
                connector = _shortest_path_inside_polygon(route[-1], stroke[0], field_polygon)
                connectors.append(connector)
                route.extend(connector[1:])
            elif not route:
                # The scan endpoint on the configured entry edge is the actual
                # anchor. The UAV approaches it directly.
                route.append(stroke[0])
            if route[-1] != stroke[0]:
                route.append(stroke[0])
            route.append(stroke[1])
            ordered_strokes.append(stroke)
            current = stroke[1]
    return SweepPlan(_remove_consecutive_duplicates(route), ordered_strokes, connectors)


def generate_path(
    coverage_region: Sequence[GridPoint],
    orientation: float,
    coverage_start_point: Point2D,
    *,
    lane_spacing: float | None = None,
    field_polygon: BaseGeometry | None = None,
    target_boundary=None,
    entry_boundary=None,
) -> list[Point2D]:
    """Generate the coverage-only route with the requested scan anchor first.

    ``coverage_start_point`` must be a point on one of the candidate scan
    endpoints. The executor obtains it from a provisional sweep of this region,
    then calls this function to make the anchor an explicit planner input.
    """
    path = generate_boustrophedon_path(
        coverage_region,
        orientation,
        start_point=coverage_start_point,
        coverage_start_point=coverage_start_point,
        lane_spacing=lane_spacing,
        field_polygon=field_polygon,
        target_boundary=target_boundary,
        entry_boundary=entry_boundary,
    )
    if not path:
        return []
    if path[0] != coverage_start_point:
        path = [coverage_start_point, *path]
    return path


def _shortest_path_inside_polygon(
    start: Point2D,
    end: Point2D,
    polygon: BaseGeometry | None,
) -> list[Point2D]:
    """Find a shortest visibility-graph connector inside the field polygon."""
    if start == end:
        return [start]
    direct = LineString([start, end])
    if polygon is None or polygon.buffer(1e-7).covers(direct):
        return [start, end]

    vertices: list[Point2D] = [start, end]
    geometries = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
    for geometry in geometries:
        if not hasattr(geometry, "exterior"):
            continue
        vertices.extend((float(x), float(y)) for x, y in list(geometry.exterior.coords)[:-1])
        for ring in geometry.interiors:
            vertices.extend((float(x), float(y)) for x, y in list(ring.coords)[:-1])

    visible = polygon.buffer(1e-7)
    graph: list[list[tuple[float, int]]] = [[] for _ in vertices]
    for first in range(len(vertices)):
        for second in range(first + 1, len(vertices)):
            segment = LineString([vertices[first], vertices[second]])
            if visible.covers(segment):
                graph[first].append((segment.length, second))
                graph[second].append((segment.length, first))

    costs = [float("inf")] * len(vertices)
    previous = [-1] * len(vertices)
    costs[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        cost, node = heappop(queue)
        if cost != costs[node]:
            continue
        if node == 1:
            break
        for edge_cost, neighbor in graph[node]:
            candidate = cost + edge_cost
            if candidate < costs[neighbor]:
                costs[neighbor] = candidate
                previous[neighbor] = node
                heappush(queue, (candidate, neighbor))
    if costs[1] == float("inf"):
        return [start, end]
    indices = []
    node = 1
    while node >= 0:
        indices.append(node)
        if node == 0:
            break
        node = previous[node]
    return [vertices[index] for index in reversed(indices)]


def _line_parts(geometry):
    """Yield line components from a Shapely intersection result."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "LineString":
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _line_parts(part)


def _from_scan_coordinates(u: float, v: float, orientation: float) -> Point2D:
    """Rotate one scan-frame coordinate back into field-local XY."""
    theta = radians(orientation)
    return (
        u * cos(theta) - v * sin(theta),
        u * sin(theta) + v * cos(theta),
    )


def _remove_consecutive_duplicates(points: Sequence[Point2D]) -> list[Point2D]:
    """Drop repeated waypoints produced by touching scanline intervals."""
    result: list[Point2D] = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    return result
