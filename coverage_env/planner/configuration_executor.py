"""Instantiate and execute one high-level coverage configuration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot, isfinite
from typing import TYPE_CHECKING

from shapely.geometry import LineString, Point

from coverage_env.geometry.discretization import GridPoint, discretize_field
from coverage_env.planner.configuration import CoverageConfiguration
from coverage_env.planner.boundary_aware_sweep import BoundaryAwareSweepPlanner
from coverage_env.planner.boundary_region_generator import BoundaryRegionGenerator

if TYPE_CHECKING:
    from coverage_env.env.environment import Environment

Point2D = tuple[float, float]


@dataclass(frozen=True)
class CoverageExecutionResult:
    """Measured result of one configuration, with distinct transit/sweep legs."""

    region: list[GridPoint]
    entry_boundary_point: Point2D
    coverage_start_point: Point2D
    exit_boundary_point: Point2D
    entry_transit_path: list[Point2D]
    coverage_path: list[Point2D]
    coverage_strokes: list[tuple[Point2D, Point2D]]
    scan_transit_paths: list[list[Point2D]]
    exit_transit_path: list[Point2D]
    total_path: list[Point2D]
    distance: float
    energy: float
    covered_area: float
    field_id: str | int
    uav_id: str | int
    orientation: float
    reachable_component: list[GridPoint]
    scan_order: list[GridPoint]
    growth_edge_id: str
    max_reachable_area: float
    remaining_points: list[GridPoint]

    @property
    def entry_point(self) -> Point2D:
        """Backward-compatible alias for the boundary entry point."""
        return self.entry_boundary_point

    @property
    def exit_point(self) -> Point2D:
        """Backward-compatible alias for the boundary exit point."""
        return self.exit_boundary_point

    @property
    def energy_consumption(self) -> float:
        """Alias that makes the measured energy field self-documenting."""
        return self.energy


class ConfigurationExecutor:
    """Resolve entry, reachable coverage, anchor, exit, and execute their route."""

    def __init__(self, grid_resolution: float = 5.0) -> None:
        """Create an executor; resolution is used if a field has no grid yet."""
        if not isfinite(grid_resolution) or grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be a finite number greater than zero.")
        self.grid_resolution = float(grid_resolution)
        self.region_generator = BoundaryRegionGenerator()
        self.path_planner = BoundaryAwareSweepPlanner()

    def execute(
        self,
        configuration: CoverageConfiguration,
        environment: "Environment",
    ) -> CoverageExecutionResult:
        """Execute a configuration and charge all three route segments."""
        if not isinstance(configuration, CoverageConfiguration):
            raise TypeError("configuration must be a CoverageConfiguration.")
        if not environment.uavs:
            raise ValueError("Environment must contain at least one UAV.")

        field = environment.get_field(configuration.field_id)
        uav = environment.uavs[0]
        if uav.current_position is None:
            uav.reset(environment.depot.get_position())
        uav_start = tuple(uav.current_position)
        if not field.grid_points:
            discretize_field(field, self.grid_resolution)

        entry_boundary = field.get_boundary(configuration.entry_boundary_id)
        exit_boundary = field.get_boundary(configuration.exit_boundary_id)
        remaining_area = sum(
            point.point_area for point in field.grid_points if not point.covered
        )
        selection = self.region_generator.generate(
            field,
            entry_boundary,
            exit_boundary,
            configuration.workload_ratio,
            workload_reference_area=remaining_area,
        )
        region = selection.region
        sweep_plan = self.path_planner.plan(
            region,
            field,
            entry_boundary,
            exit_boundary,
            current_position=uav_start,
            lane_spacing=uav.coverage_width,
        )
        entry_boundary_point = sweep_plan.entry_point
        coverage_path = sweep_plan.coverage_path
        if not coverage_path:
            raise ValueError("Coverage path planner returned no points.")
        coverage_start_point = coverage_path[0]
        exit_boundary_point = sweep_plan.exit_point
        entry_connector = self._inside_field_connector(
            entry_boundary_point, coverage_start_point, field.xy_polygon
        )
        exit_connector = self._inside_field_connector(
            coverage_path[-1], exit_boundary_point, field.xy_polygon
        )
        entry_transit_path = self._dedupe_path(
            [uav_start, entry_boundary_point, *entry_connector[1:]]
        )
        exit_transit_path = self._dedupe_path(exit_connector)
        total_path = self._join_paths(
            entry_transit_path, coverage_path, exit_transit_path
        )

        covered_area = float(sum(point.point_area for point in region))
        feedback = uav.execute_path(total_path, area=covered_area)
        for point in region:
            point.covered = True
        remaining_points = field.get_uncovered_points()
        if field.field_id not in uav.visited_fields:
            uav.visited_fields.append(field.field_id)

        return CoverageExecutionResult(
            region=region,
            entry_boundary_point=entry_boundary_point,
            coverage_start_point=coverage_start_point,
            exit_boundary_point=exit_boundary_point,
            entry_transit_path=entry_transit_path,
            coverage_path=coverage_path,
            coverage_strokes=sweep_plan.coverage_strokes,
            scan_transit_paths=sweep_plan.scan_transit_paths,
            exit_transit_path=exit_transit_path,
            total_path=total_path,
            distance=float(feedback["path_length"]),
            energy=float(feedback["energy_consumption"]),
            covered_area=covered_area,
            field_id=field.field_id,
            uav_id=uav.uav_id,
            orientation=sweep_plan.orientation,
            reachable_component=selection.scan_order,
            scan_order=selection.scan_order,
            growth_edge_id=selection.growth_edge.id,
            max_reachable_area=selection.max_reachable_area,
            remaining_points=remaining_points,
        )

    @staticmethod
    def _project_to_boundary(boundary, reference: Point2D) -> Point2D:
        """Return the exact closest point on a straight boundary segment."""
        segment = LineString([boundary.start_point, boundary.end_point])
        projected = segment.interpolate(segment.project(Point(reference)))
        return float(projected.x), float(projected.y)

    @staticmethod
    def _inside_field_connector(start: Point2D, end: Point2D, polygon) -> list[Point2D]:
        """Return the direct or visibility-graph connector contained in Field."""
        if start == end:
            return [start]
        line = LineString([start, end])
        if polygon.buffer(1e-7).covers(line):
            return [start, end]
        from coverage_env.planner.coverage_path import _shortest_path_inside_polygon

        path = _shortest_path_inside_polygon(start, end, polygon)
        if not polygon.buffer(1e-5).covers(LineString(path)):
            raise ValueError("Could not connect a scan endpoint to its boundary inside the field.")
        return path

    @staticmethod
    def _entry_reachable_component(
        points: list[GridPoint], entry_point: Point2D
    ) -> list[GridPoint]:
        """Build the uncovered 8-neighbor component seeded nearest the entry."""
        available = [point for point in points if not point.covered]
        if not available:
            raise ValueError("Field has no uncovered grid points.")
        by_id = {point.id: point for point in available}
        root = min(
            available,
            key=lambda point: hypot(point.x - entry_point[0], point.y - entry_point[1]),
        )
        reached: set[str] = set()
        pending = deque([root.id])
        while pending:
            point_id = pending.popleft()
            if point_id in reached or point_id not in by_id:
                continue
            reached.add(point_id)
            pending.extend(by_id[point_id].neighbors)
        return [point for point in available if point.id in reached]

    @staticmethod
    def _dedupe_path(path: list[Point2D]) -> list[Point2D]:
        """Remove adjacent duplicate waypoints without reordering the route."""
        result: list[Point2D] = []
        for point in path:
            if not result or point != result[-1]:
                result.append(point)
        return result

    @classmethod
    def _join_paths(cls, *paths: list[Point2D]) -> list[Point2D]:
        """Concatenate path segments while retaining one copy of joins."""
        result: list[Point2D] = []
        for path in paths:
            if not path:
                continue
            if result and result[-1] == path[0]:
                result.extend(path[1:])
            else:
                result.extend(path)
        return cls._dedupe_path(result)
