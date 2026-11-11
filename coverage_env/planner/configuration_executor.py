"""Instantiate and execute one high-level coverage configuration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from copy import deepcopy
from types import SimpleNamespace

from coverage_env.planner.coverage_plan import CoveragePlan, fingerprints
from coverage_env.planner.framework_contracts import RegionTask
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
    payload: float
    transfer_distance: float
    coverage_distance: float
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


class InsufficientResources(ValueError):
    """A valid plan cannot be executed with the available resources."""


class ConfigurationExecutor:
    """Resolve entry, reachable coverage, anchor, exit, and execute their route."""

    def __init__(self, grid_resolution: float = 5.0) -> None:
        """Create an executor; resolution is used if a field has no grid yet."""
        if not isfinite(grid_resolution) or grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be a finite number greater than zero.")
        self.grid_resolution = float(grid_resolution)
        self.region_generator = BoundaryRegionGenerator()
        self.path_planner = BoundaryAwareSweepPlanner()

    def execute(self, configuration, environment) -> CoverageExecutionResult:
        """Legacy CC wrapper; retain lazy initialization, then plan and execute."""
        if not isinstance(configuration, CoverageConfiguration):
            raise TypeError("configuration must be a CoverageConfiguration.")
        if not environment.uavs:
            raise ValueError("Environment must contain at least one UAV.")
        field = environment.get_field(configuration.field_id)
        if environment.uavs[0].current_position is None:
            environment.uavs[0].reset(environment.depot.get_position())
        if not field.grid_points:
            discretize_field(field, self.grid_resolution)
        return self.execute_plan(self.plan_configuration(configuration, environment), environment)

    def plan_configuration(self, configuration, environment) -> CoveragePlan:
        """Pure CC planning on an initialized snapshot; no physical mutation."""
        if not isinstance(configuration, CoverageConfiguration):
            raise TypeError("Expected CoverageConfiguration")
        field, uav = self._planning_context(configuration.field_id, environment)
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
        return self._plan_region(configuration, environment, selection.region, selection,
                                 source="cc", task_id="cc")

    @staticmethod
    def _planning_context(field_id, environment):
        if not environment.uavs:
            raise ValueError("Planning requires an initialized UAV")
        field = environment.get_field(field_id)
        uav = environment.uavs[0]
        if not field.grid_points or uav.current_position is None:
            raise ValueError("Initialize/discretize the environment before planning")
        return field, uav

    def plan_region(self, task: RegionTask, entry_boundary_id, exit_boundary_id, environment):
        """Honor exactly the supplied region; never invoke CC region generation."""
        if not isinstance(task, RegionTask):
            raise TypeError("Expected RegionTask")
        if len(environment.uavs) != 1:
            raise ValueError("Explicit region tasks require exactly one UAV")
        field, _ = self._planning_context(task.field_id, environment)
        by_id = {p.id: p for p in field.grid_points}
        if any(pid not in by_id or by_id[pid].covered for pid in task.point_ids):
            raise ValueError("Region contains unknown or already covered points")
        region = [by_id[pid] for pid in task.point_ids]
        gates = SimpleNamespace(field_id=task.field_id, entry_boundary_id=entry_boundary_id,
                                exit_boundary_id=exit_boundary_id)
        metadata = SimpleNamespace(scan_order=region, growth_edge=SimpleNamespace(id="explicit"),
                                   max_reachable_area=sum(p.point_area for p in region))
        plan = self._plan_region(gates, environment, region, metadata,
                                 source="explicit_region", task_id=task.task_id)
        self.validate_plan(plan, environment, reserve_return=False, check_resources=False)
        return plan

    def _plan_region(self, configuration, environment, region, selection, *, source, task_id):
        field, uav = self._planning_context(configuration.field_id, environment)
        uav_start = tuple(uav.current_position)
        entry_boundary = field.get_boundary(configuration.entry_boundary_id)
        exit_boundary = field.get_boundary(configuration.exit_boundary_id)
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

        instance, config, state = fingerprints(environment, field)
        return CoveragePlan(
            1, source, task_id, field.field_id, configuration.entry_boundary_id,
            configuration.exit_boundary_id, tuple(p.id for p in region), tuple(entry_boundary_point),
            tuple(entry_transit_path), tuple(coverage_path), tuple(exit_transit_path),
            tuple(tuple(stroke) for stroke in sweep_plan.coverage_strokes),
            tuple(tuple(path) for path in sweep_plan.scan_transit_paths), sweep_plan.orientation,
            tuple(p.id for p in selection.scan_order), selection.growth_edge.id,
            selection.max_reachable_area, instance, config, state,
        )

    def validate_plan(self, plan, environment, *, reserve_return=True, check_resources=True):
        if not isinstance(plan, CoveragePlan) or plan.schema_version != 1:
            raise ValueError("Unsupported coverage plan")
        if plan.source not in ('cc', 'explicit_region'):
            raise ValueError("Unknown plan source")
        field, uav = self._planning_context(plan.field_id, environment)
        if fingerprints(environment, field) != (plan.instance_fingerprint,
                plan.config_fingerprint, plan.state_fingerprint):
            raise ValueError("Stale plan or instance/configuration mismatch")
        by_id = {p.id: p for p in field.grid_points}
        if not plan.point_ids or len(set(plan.point_ids)) != len(plan.point_ids):
            raise ValueError("Empty or duplicate planned region")
        if any(pid not in by_id or by_id[pid].covered for pid in plan.point_ids):
            raise ValueError("Unknown or already covered planned point")
        paths = (plan.entry_transit_path, plan.coverage_path, plan.exit_transit_path)
        if any(not path for path in paths):
            raise ValueError("Empty planned path")
        for path in (*paths, *plan.scan_transit_paths, *plan.coverage_strokes):
            if any(len(p) != 2 or not all(isfinite(v) for v in p) for p in path):
                raise ValueError("Nonfinite planned geometry")
        if (paths[0][0] != uav.current_position or paths[0][-1] != paths[1][0]
                or paths[1][-1] != paths[2][0]):
            raise ValueError("Disconnected planned paths")
        # Explicit plans must certify coverage from spraying strokes, not their IDs.
        if plan.source == "explicit_region":
            from shapely.ops import unary_union
            strokes = [LineString(stroke) for stroke in plan.coverage_strokes
                       if len(stroke) == 2 and stroke[0] != stroke[1]]
            if not strokes:
                raise ValueError("No spraying strokes")
            # Reconstruct in-flight traversal from ordered strokes/connectors.
            route = [plan.coverage_strokes[0][0]]
            connectors = iter(plan.scan_transit_paths)
            for start, end in plan.coverage_strokes:
                if route[-1] != start:
                    connector = next(connectors, None)
                    if not connector or connector[0] != route[-1] or connector[-1] != start:
                        raise ValueError("Invalid scan connector")
                    route.extend(connector[1:])
                if route[-1] != end:
                    route.append(end)
            if next(connectors, None) is not None or tuple(self._dedupe_path(route)) != plan.coverage_path:
                raise ValueError("Coverage decomposition does not match flown path")
            sweep = unary_union(strokes)
            for pid in plan.point_ids:
                if sweep.distance(Point(by_id[pid].coordinate)) > uav.coverage_width / 2 + 1e-5:
                    raise ValueError("Planned strokes do not cover assigned grid points")
            flown = LineString(plan.coverage_path) if len(plan.coverage_path) > 1 else Point(plan.coverage_path[0])
            if any(not flown.buffer(1e-6).covers(stroke) for stroke in strokes):
                raise ValueError("Spraying stroke absent from executed coverage path")
            entry_index = plan.entry_transit_path.index(plan.entry_boundary_point)
            for path in (plan.coverage_path, plan.exit_transit_path, plan.entry_transit_path[entry_index:]):
                if len(path) > 1 and not field.xy_polygon.buffer(1e-5).covers(LineString(path)):
                    raise ValueError("In-field path leaves field")
            for gate_id, point in ((plan.entry_boundary_id, plan.entry_boundary_point),
                                   (plan.exit_boundary_id, plan.exit_transit_path[-1])):
                gate = field.get_boundary(gate_id)
                if LineString([gate.start_point, gate.end_point]).distance(Point(point)) > 1e-5:
                    raise ValueError("Planned endpoint is not on declared boundary")
        area = sum(by_id[pid].point_area for pid in plan.point_ids)
        distance = sum(self._path_length(path) for path in paths)
        energy = uav.calculate_energy(distance, area)
        if reserve_return:
            energy += uav.alpha * hypot(plan.exit_transit_path[-1][0]-environment.depot.position[0],
                                       plan.exit_transit_path[-1][1]-environment.depot.position[1])
        if check_resources and (energy > uav.remaining_energy + 1e-8
                or uav.payload_per_area * area > uav.remaining_payload + 1e-8):
            raise InsufficientResources("Planned route including return exceeds available resources")

    def execute_plan(self, plan, environment, *, reserve_return=False):
        """Execute stored geometry without replanning; rollback all segment mutations."""
        self.validate_plan(plan, environment,
                           reserve_return=reserve_return or plan.source == 'explicit_region')
        field = environment.get_field(plan.field_id)
        uav = environment.uavs[0]
        saved = deepcopy(uav.__dict__)
        covered = [(p, p.covered) for p in field.grid_points]
        try:
            return self._execute_validated(plan, environment)
        except Exception:
            uav.__dict__.clear()
            uav.__dict__.update(saved)
            for point, value in covered:
                point.covered = value
            raise

    def _execute_validated(self, plan, environment):
        field = environment.get_field(plan.field_id)
        uav = environment.uavs[0]
        by_id = {p.id: p for p in field.grid_points}
        region = [by_id[pid] for pid in plan.point_ids]
        entry_transit_path = list(plan.entry_transit_path)
        coverage_path = list(plan.coverage_path)
        exit_transit_path = list(plan.exit_transit_path)
        entry_boundary_point = plan.entry_boundary_point
        coverage_start_point = coverage_path[0]
        exit_boundary_point = exit_transit_path[-1]
        total_path = self._join_paths(entry_transit_path, coverage_path, exit_transit_path)
        sweep_plan = SimpleNamespace(coverage_strokes=list(plan.coverage_strokes),
                                     scan_transit_paths=[list(p) for p in plan.scan_transit_paths],
                                     orientation=plan.orientation)
        selection = SimpleNamespace(scan_order=[by_id[pid] for pid in plan.scan_order_ids],
                                    growth_edge=SimpleNamespace(id=plan.growth_edge_id),
                                    max_reachable_area=plan.max_reachable_area)
        covered_area = float(sum(point.point_area for point in region))
        entry_feedback = uav.execute_path(entry_transit_path, area=0.0)
        coverage_feedback = uav.execute_path(coverage_path, area=covered_area)
        exit_feedback = uav.execute_path(exit_transit_path, area=0.0)
        for point in region:
            point.covered = True
        remaining_points = field.get_uncovered_points()
        if field.field_id not in uav.visited_fields:
            uav.visited_fields.append(field.field_id)

        entry_distance = float(entry_feedback["path_length"])
        exit_distance = float(exit_feedback["path_length"])
        scan_transit_distance = sum(
            self._path_length(path) for path in sweep_plan.scan_transit_paths
        )
        spray_distance = sum(
            hypot(end[0] - start[0], end[1] - start[1])
            for start, end in sweep_plan.coverage_strokes
        )
        distance = entry_distance + float(coverage_feedback["path_length"]) + exit_distance
        energy = (
            float(entry_feedback["energy_consumption"])
            + float(coverage_feedback["energy_consumption"])
            + float(exit_feedback["energy_consumption"])
        )
        payload = (
            float(entry_feedback["payload_consumption"])
            + float(coverage_feedback["payload_consumption"])
            + float(exit_feedback["payload_consumption"])
        )
        transfer_distance = entry_distance + scan_transit_distance + exit_distance

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
            distance=distance,
            energy=energy,
            payload=payload,
            transfer_distance=transfer_distance,
            coverage_distance=spray_distance,
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

    @staticmethod
    def _path_length(path: list[Point2D]) -> float:
        """Return the Euclidean length of a waypoint sequence."""
        return sum(
            hypot(end[0] - start[0], end[1] - start[1])
            for start, end in zip(path[:-1], path[1:])
        )

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
