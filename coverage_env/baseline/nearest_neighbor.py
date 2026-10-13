"""Nearest-unfinished-field heuristic used as a validation baseline."""

from __future__ import annotations

from math import ceil, hypot, nextafter, sqrt

from shapely.geometry import LineString, Point

from coverage_env.planner.configuration import CoverageConfiguration
from coverage_env.rl.action_space import CoverageAction


class NearestNeighborAgent:
    """Choose the nearest unfinished field and an energy-feasible workload."""

    def __init__(self, environment, config: dict) -> None:
        """Bind the baseline to an environment and its unified config."""
        self.environment = environment
        self.config = config
        self.last_failure: str | None = None

    def select_action(self, state: dict) -> CoverageConfiguration | None:
        """Return one configuration, or ``None`` if done/no safe action exists."""
        self.last_failure = None
        unfinished = [
            field for field in self.environment.fields
            if state["fields"][field.field_id]["uncovered_points"] > 0
        ]
        if not unfinished:
            return None

        uav = self.environment.uavs[0]
        if uav.current_position is None:
            self.last_failure = "UAV position is not initialized."
            return None
        position = uav.current_position
        # Prefer the nearest field, but skip one whose current uncovered
        # component cannot be reached through the selected growth edge.  The
        # executor's edge-guided region can be smaller than the component
        # reachable from the entry boundary alone, so use the environment's
        # geometry-aware estimate as the workload ceiling.
        unfinished.sort(key=lambda item: hypot(
            item.centroid[0] - position[0], item.centroid[1] - position[1]
        ))
        field = entry_boundary = None
        reachable_area = environment_workload = 0.0
        for candidate_field in unfinished:
            candidate_entry = min(
                candidate_field.boundaries,
                key=lambda boundary: LineString(
                    [boundary.start_point, boundary.end_point]
                ).distance(Point(position)),
            )
            estimated = self.environment.estimate_workload(CoverageAction(
                candidate_field.field_id, candidate_entry.id, candidate_entry.id
            ), state)
            if estimated > 0.0:
                field = candidate_field
                entry_boundary = candidate_entry
                environment_workload = estimated
                reachable_area = self._entry_connected_area(
                    candidate_field, candidate_entry, position
                )
                break
        if field is None or entry_boundary is None:
            self.last_failure = (
                "No unfinished field has a positive workload reachable from "
                "its selected entry/growth boundary."
            )
            return None

        coverage_config = self.config["coverage"]
        energy_config = self.config["uav"]["energy"]
        remaining_area = float(state["fields"][field.field_id]["remaining_area"])
        if remaining_area <= 0.0:
            self.last_failure = "Field has uncovered points but no remaining area."
            return None

        span = max(
            field.bounding_box[2] - field.bounding_box[0],
            field.bounding_box[3] - field.bounding_box[1],
            uav.coverage_width,
        )
        diagonal = sqrt(remaining_area)
        approach_distance = hypot(
            field.centroid[0] - position[0], field.centroid[1] - position[1]
        ) + 0.5 * diagonal
        sweep_distance_per_full_workload = (
            remaining_area / uav.coverage_width
            + remaining_area / span
            + diagonal
        )
        return_distance = hypot(
            field.centroid[0] - self.environment.depot.position[0],
            field.centroid[1] - self.environment.depot.position[1],
        ) + 0.5 * diagonal
        return_energy = uav.alpha * return_distance
        reserve_energy = (
            float(energy_config["energy_margin"]) * uav.max_energy
        )
        fixed_energy = uav.alpha * approach_distance
        variable_energy = (
            uav.alpha * sweep_distance_per_full_workload
            + uav.beta * remaining_area
        )
        safe_budget = uav.remaining_energy - return_energy - reserve_energy
        if variable_energy <= 0.0:
            self.last_failure = "Estimated workload energy is not positive."
            return None
        feasible_ratio = (safe_budget - fixed_energy) / variable_energy
        connected_ratio = min(
            1.0,
            nextafter(reachable_area / remaining_area, 0.0)
            if reachable_area < remaining_area else 1.0,
        )
        minimum = float(coverage_config["min_workload"])
        if feasible_ratio < minimum:
            self.last_failure = (
                "Insufficient energy for minimum workload while preserving "
                "estimated return energy and reserve."
            )
            return None
        ratio = min(
            float(coverage_config["default_workload"]),
            float(coverage_config["max_workload"]),
            feasible_ratio,
            connected_ratio,
            environment_workload,
        )
        # The growth-edge estimate is a hard geometry limit. Preserve a smaller
        # positive ratio when fragmentation leaves less than the preferred
        # minimum; never raise the request above either reachability estimate.
        if ratio <= 0.0:
            self.last_failure = "No uncovered component is reachable from the entry boundary."
            return None
        return CoverageConfiguration(
            field_id=field.field_id,
            entry_boundary_id=entry_boundary.id,
            exit_boundary_id=entry_boundary.id,
            workload_ratio=ratio,
        )

    @staticmethod
    def _entry_connected_area(field, entry_boundary, position) -> float:
        """Estimate the uncovered component seeded by the chosen entry edge."""
        resolution = float(field.grid_resolution or 5.0)
        sample_count = max(1, ceil(entry_boundary.length / resolution))
        start_x, start_y = entry_boundary.start_point
        end_x, end_y = entry_boundary.end_point
        samples = [
            (
                start_x + (end_x - start_x) * index / sample_count,
                start_y + (end_y - start_y) * index / sample_count,
            )
            for index in range(sample_count + 1)
        ]
        entry_reference = min(
            samples,
            key=lambda point: hypot(point[0] - position[0], point[1] - position[1]),
        )
        available = field.get_uncovered_points()
        if not available:
            return 0.0
        root = min(
            available,
            key=lambda point: hypot(point.x - entry_reference[0], point.y - entry_reference[1]),
        )
        available_by_id = {point.id: point for point in available}
        reached = set()
        pending = [root.id]
        while pending:
            point_id = pending.pop()
            if point_id in reached or point_id not in available_by_id:
                continue
            reached.add(point_id)
            pending.extend(available_by_id[point_id].neighbors)
        return float(sum(available_by_id[point_id].point_area for point_id in reached))
