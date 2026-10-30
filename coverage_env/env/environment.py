"""Geometry-only field registry for the Phase 1 environment foundation."""

from __future__ import annotations

from coverage_env.env.time_model import heading_from_path, path_time

import random
from collections import deque
from copy import deepcopy
from typing import Dict, Iterable, Optional, Union

from coverage_env.env.depot import Depot
from coverage_env.env.episode_logger import EpisodeLogger
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV


class EnvironmentStep(tuple):
    """Four-item Gym transition with legacy feedback-key access.

    New code should unpack ``next_state, reward, done, info``. Existing mission
    examples that read ``result["coverage"]`` continue to work during migration.
    """

    def __new__(cls, next_state, reward, done, info):
        return super().__new__(cls, (next_state, reward, done, info))

    def __getitem__(self, key):
        if isinstance(key, str):
            next_state, _reward, _done, info = tuple.__getitem__(self, slice(None))
            legacy = {
                "state": next_state, "new_state": next_state,
                "distance": info.get("path_length", 0.0),
                "energy": info.get("energy", 0.0),
                "coverage": info.get("coverage_gain", 0.0),
                "exit_point": info.get("exit_point"),
                "path": info.get("path", []),
            }
            return info[key] if key in info else legacy[key]
        return tuple.__getitem__(self, key)

    def get(self, key, default=None):
        """Provide mapping-style lookup for old feedback consumers."""
        try:
            return self[key]
        except KeyError:
            return default


class FieldEnvironment:
    """Own a collection of fields and provide stable id-based lookup."""

    def __init__(self, fields: Iterable[Field] = ()) -> None:
        """Initialize the registry with zero or more field objects."""
        self._fields: Dict[Union[str, int], Field] = {}
        for field in fields:
            self.add_field(field)

    @property
    def fields(self) -> tuple[Field, ...]:
        """Return registered fields in insertion order."""
        return tuple(self._fields.values())

    def add_field(self, field: Field) -> None:
        """Register a field, rejecting duplicate ids."""
        if field.field_id in self._fields:
            raise ValueError(f"Duplicate field id: {field.field_id!r}.")
        self._fields[field.field_id] = field

    def get_field(self, field_id: Union[str, int]) -> Field:
        """Return a registered field by id."""
        try:
            return self._fields[field_id]
        except KeyError as exc:
            raise KeyError(f"No field registered with id {field_id!r}.") from exc


class CoverageEnvironment(FieldEnvironment):
    """Multi-field task manager with action-level coverage execution.

    ``step_configuration`` is the simulator transition interface that future
    policies can call with a state-conditioned high-level coverage action.
    """

    def __init__(
        self,
        fields: Iterable[Field],
        depot: Depot,
        uavs: Iterable[UAV] = (),
        *,
        rl_config: dict | None = None,
        config: dict | None = None,
        rng: Optional[random.Random] = None,
        randomize_each_reset: bool = False,
        randomize_uav: bool = True,
    ) -> None:
        """Create an environment with fields, a depot, and zero or more UAVs.

        Supplying ``config`` and ``rng`` enables cross-episode generalization:
        when ``randomize_each_reset`` is true, ``reset()`` resamples the fields
        (count and geometry), the depot, and the UAV capability parameters from
        config using the environment-owned ``random.Random`` (so a seed
        reproduces the whole episode sequence).
        """
        self._heading_valid: dict = {}
        self._reset_time_ledger()
        # The flight-time model and the coverage credit use FIXED speed / width, so a
        # randomized UAV would silently make the reward inconsistent with the physical
        # model. Fail loudly instead of producing an uninterpretable training signal.
        _time_cfg = (rl_config or {}).get("time", {}) if isinstance(rl_config, dict) else {}
        _reward_cfg = (rl_config or {}).get("reward", {}) if isinstance(rl_config, dict) else {}
        # Only the TIME reward depends on these being fixed; the distance objective is
        # valid with randomized capability, so the check is scoped to that mode.
        _time_mode = str((_reward_cfg.get("mode") or "distance")).lower() == "time"
        for _uav in uavs or ():
            if not _time_mode:
                break
            if _time_cfg.get("cruise_speed") is not None:
                if abs(float(_uav.speed) - float(_time_cfg["cruise_speed"])) > 1e-9:
                    raise ValueError(
                        f"UAV speed {_uav.speed} != time.cruise_speed "
                        f"{_time_cfg['cruise_speed']}; the time model assumes a fixed speed")
            if _reward_cfg.get("coverage_width") is not None:
                if abs(float(_uav.coverage_width) - float(_reward_cfg["coverage_width"])) > 1e-9:
                    raise ValueError(
                        f"UAV coverage_width {_uav.coverage_width} != reward.coverage_width "
                        f"{_reward_cfg['coverage_width']}; the coverage credit assumes a fixed width")
        # Cumulative flight-time ledger, filled on every step and reset with the
        # episode. summary["total_time"] is this sum, so evaluation and checkpoint
        # selection never have to re-add per-step fields.
        self._time_totals = {"work": 0.0, "nonwork": 0.0, "maneuver": 0.0,
                             "service": 0.0, "resupply_or_return": 0.0, "total": 0.0}
        super().__init__(fields)
        if not isinstance(depot, Depot):
            raise TypeError("depot must be a Depot instance.")
        self.depot = depot
        self.uavs = list(uavs)
        if not all(isinstance(uav, UAV) for uav in self.uavs):
            raise TypeError("Every item in uavs must be a UAV instance.")
        self._align_field_coordinate_frames()
        self._execution_records: list[dict] = []
        self._mission_trajectories: dict = {uav.uav_id: [] for uav in self.uavs}
        self.episode_logger = EpisodeLogger()
        self.rl_config = dict(rl_config or {})
        self._config = config
        self._rng = rng
        self._randomize_each_reset = bool(randomize_each_reset and config is not None)
        self._randomize_uav = bool(randomize_uav)
        # The first reset keeps the constructor-provided instance; later resets
        # regenerate it when generalization is enabled.
        self._regenerate_next_reset = False
        self._instance_generator = None
        self._reachable_area_cache: dict = {}
        self._component_cache: dict = {}
        self._action_estimate_cache: dict = {}
        self._growth_selection_cache: dict = {}
        self._growth_area_cache: dict = {}
        self._consecutive_failures = 0
        self._resupply_count = 0
        self._termination_reason: str | None = None
        self.total_area = self._compute_total_area() or 1.0

    def _align_field_coordinate_frames(self) -> None:
        """Put all fields in the first field's projected local XY frame.

        Each Field initially chooses its own local origin. Reusing the first
        field's transformer prevents geographically separate fields from
        overlapping at (0, 0) in mission plots and transition distances.
        Depot coordinates are expected in that shared frame.
        """
        fields = self.fields
        if not fields:
            return
        shared_transformer = fields[0].coordinate_transformer
        for field in fields[1:]:
            field.set_coordinate_transformer(shared_transformer)

    def reset(self) -> dict:
        """Reset fields, UAVs, mission history, and return the initial state.

        When cross-episode generalization is enabled, every reset after the
        first resamples the fields (count + geometry), the depot, and the UAV
        parameters before restoring the mission state. The cached
        ``total_area`` always reflects the current episode's instance.
        """
        self._reset_time_ledger()
        self._heading_valid = {}
        if self._regenerate_next_reset:
            self._resample_instance()
        self._regenerate_next_reset = self._randomize_each_reset
        position = self.depot.get_position()
        for field in self.fields:
            if not field.grid_points:
                from coverage_env.geometry.discretization import discretize_field

                discretize_field(
                    field, float(self.rl_config.get("grid_resolution", 5.0))
                )
            for point in field.grid_points:
                point.covered = False
        self.total_area = self._compute_total_area() or 1.0
        for uav in self.uavs:
            uav.reset(position)
        self._execution_records.clear()
        self._mission_trajectories = {uav.uav_id: [] for uav in self.uavs}
        self._reachable_area_cache.clear()
        self._component_cache.clear()
        self._action_estimate_cache.clear()
        self._growth_selection_cache.clear()
        self._growth_area_cache.clear()
        self._consecutive_failures = 0
        self._resupply_count = 0
        self._termination_reason = None
        self.episode_logger.reset()
        state = self.get_state()
        # Preserve the Phase 3 reset contract (reset()[uav_id]) while returning
        # the richer complete environment observation required by this phase.
        state.update({uav.uav_id: uav.get_state() for uav in self.uavs})
        return state

    def _compute_total_area(self) -> float:
        """Return the total discretized (coverable) area of the current fields."""
        return float(sum(
            point.point_area
            for field in self.fields
            for point in field.grid_points
        ))

    def _resample_instance(self) -> None:
        """Rebuild fields, depot, and UAVs from config using the held RNG."""
        if self._config is None or self._rng is None:
            return
        from coverage_env.env.instance_generator import InstanceGenerator

        generator = self._instance_generator
        if generator is None:
            generator = InstanceGenerator(self._config)
            self._instance_generator = generator
        count = generator.sample_num_fields(self._rng)
        fields, depot, uavs, resolution = generator.sample_components(
            self._rng, count, randomize_uav=self._randomize_uav
        )
        self._replace_instance(fields, depot, uavs, resolution)

    def _replace_instance(self, fields, depot, uavs, resolution) -> None:
        """Swap in a freshly sampled instance and rebuild all derived state."""
        from coverage_env.geometry.discretization import discretize_field

        self._fields.clear()
        for field in fields:
            self.add_field(field)
        self.depot = depot
        self.uavs = list(uavs)
        self._align_field_coordinate_frames()
        self._mission_trajectories = {uav.uav_id: [] for uav in self.uavs}
        self._execution_records.clear()
        self._reachable_area_cache.clear()
        self._component_cache.clear()
        self._action_estimate_cache.clear()
        self._growth_selection_cache.clear()
        self._growth_area_cache.clear()
        self._configuration_executor = None
        for field in self.fields:
            discretize_field(field, resolution)

    def _reset_time_ledger(self) -> None:
        """Zero the cumulative flight-time ledger (called from __init__ and reset)."""
        self._time_totals = {"work": 0.0, "nonwork": 0.0, "maneuver": 0.0,
                             "service": 0.0, "resupply_or_return": 0.0, "total": 0.0}

    def get_state(self) -> dict:
        """Return UAV, field, and depot state as an ordinary dictionary."""
        field_states = {}
        reference_position = (
            self.uavs[0].current_position
            if self.uavs and self.uavs[0].current_position is not None
            else self.depot.get_position()
        )
        for field in self.fields:
            total_discrete_area = sum(point.point_area for point in field.grid_points)
            covered_area = field.calculate_coverage_area()
            denominator = total_discrete_area if total_discrete_area > 0 else field.area
            coverage_ratio = min(1.0, covered_area / denominator) if denominator else 0.0
            uncovered = field.get_uncovered_points()
            coverage_key = (field.field_id, tuple(point.id for point in uncovered))
            component_data = self._component_cache.get(coverage_key)
            if component_data is None:
                from shapely.geometry import Point
                from shapely.strtree import STRtree

                point_by_id = {point.id: point for point in uncovered}
                component_by_id = {}
                component_areas = {}
                for point in uncovered:
                    if point.id in component_by_id:
                        continue
                    component_id = point.id
                    pending = deque([point.id])
                    component_by_id[point.id] = component_id
                    component_area = 0.0
                    while pending:
                        point_id = pending.popleft()
                        component_area += point_by_id[point_id].point_area
                        for neighbor_id in point_by_id[point_id].neighbors:
                            if neighbor_id in point_by_id and neighbor_id not in component_by_id:
                                component_by_id[neighbor_id] = component_id
                                pending.append(neighbor_id)
                    component_areas[component_id] = component_area
                point_geometries = [Point(point.x, point.y) for point in uncovered]
                point_tree = STRtree(point_geometries) if point_geometries else None
                component_data = (
                    component_by_id, component_areas, point_tree, uncovered
                )
                self._component_cache[coverage_key] = component_data
            component_by_id, component_areas, point_tree, component_points = component_data
            cache_key = (coverage_key, tuple(reference_position))
            reachable_by_boundary = self._reachable_area_cache.get(cache_key)
            if reachable_by_boundary is None:
                reachable_by_boundary = {}
                if component_points:
                    from shapely.geometry import LineString, Point

                    for boundary in field.boundaries:
                        segment = LineString([boundary.start_point, boundary.end_point])
                        projected = segment.interpolate(segment.project(Point(reference_position)))
                        nearest_index = int(point_tree.nearest(Point(projected.x, projected.y)))
                        root = component_points[nearest_index]
                        reachable_by_boundary[boundary.id] = component_areas.get(
                            component_by_id.get(root.id), 0.0
                        )
                else:
                    reachable_by_boundary = {boundary.id: 0.0 for boundary in field.boundaries}
                self._reachable_area_cache[cache_key] = reachable_by_boundary
            field_states[field.field_id] = {
                "field_id": field.field_id,
                "area": field.area,
                "covered_area": covered_area,
                "remaining_area": max(0.0, total_discrete_area - covered_area),
                "coverage_ratio": coverage_ratio,
                "uncovered_points": len(field.get_uncovered_points()),
                "center": field.centroid,
                "orientation": field.orientation,
                "reachable_area_by_boundary": reachable_by_boundary,
            }
        uav_states = {}
        for uav in self.uavs:
            state = uav.get_state()
            # Friendly aliases used by compact Gym-style observations.
            state["energy"] = state["remaining_energy"]
            state["distance"] = state["total_distance"]
            state["alpha"] = uav.alpha
            state["beta"] = uav.beta
            # Must be mirrored here: the UAV feature vector includes it, and replay
            # rebuilds that vector from the stored state, so omitting it silently
            # made the rollout and replay encodings disagree.
            state["payload_per_area"] = uav.payload_per_area
            state["coverage_width"] = uav.coverage_width
            # Flight direction from the flown trajectory's last segment. The maneuver
            # proxy bills the junction "previous valid segment -> next segment", so
            # the policy must observe it or the cost depends on hidden state.
            from coverage_env.env.time_model import heading_from_path
            heading = heading_from_path(self._mission_trajectories.get(uav.uav_id, []))
            valid = bool(heading is not None and self._heading_valid.get(uav.uav_id, False))
            state["heading"] = heading if valid else None
            state["heading_valid"] = valid
            uav_states[uav.uav_id] = state
        return {
            "uavs": uav_states,
            "fields": field_states,
            "boundaries": [
                {
                    "field_id": field.field_id,
                    "boundary_id": boundary.id,
                    "length": boundary.length,
                    "start_point": boundary.start_point,
                    "end_point": boundary.end_point,
                    "center": boundary.center,
                    "available": bool(field.get_uncovered_points()),
                }
                for field in self.fields
                for boundary in field.boundaries
            ],
            "depot": {
                "depot_id": self.depot.depot_id,
                "position": self.depot.get_position(),
            },
        }

    def step_configuration(self, configuration) -> dict:
        """Execute one high-level action and return transition feedback.

        The executor is imported lazily to keep the environment's geometry
        layer independent from planner modules during package initialization.
        """
        from coverage_env.planner.configuration_executor import ConfigurationExecutor

        before_state = self.get_state()
        executor = getattr(self, "_configuration_executor", None)
        if executor is None:
            executor = ConfigurationExecutor()
            self._configuration_executor = executor
        result = executor.execute(configuration, self)
        self._execution_records.append({
            "configuration": configuration,
            "field_id": result.field_id,
            "region": list(result.region),
            "path": list(result.total_path),
            "entry_point": result.entry_boundary_point,
            "exit_point": result.exit_boundary_point,
            "entry_boundary_point": result.entry_boundary_point,
            "entry_boundary_id": configuration.entry_boundary_id,
            "exit_boundary_id": configuration.exit_boundary_id,
            "coverage_start_point": result.coverage_start_point,
            "exit_boundary_point": result.exit_boundary_point,
            "entry_transit_path": list(result.entry_transit_path),
            "coverage_path": list(result.coverage_path),
            "coverage_strokes": list(result.coverage_strokes),
            "scan_transit_paths": [list(path) for path in result.scan_transit_paths],
            "exit_transit_path": list(result.exit_transit_path),
            "growth_edge_id": result.growth_edge_id,
            "max_reachable_area": result.max_reachable_area,
            "remaining_points": list(result.remaining_points),
            "total_path": list(result.total_path),
            "distance": result.distance,
            "energy": result.energy,
            "payload": result.payload,
            "transfer_distance": result.transfer_distance,
            "coverage_distance": result.coverage_distance,
            "coverage": result.covered_area,
            "orientation": result.orientation,
            "uav_id": result.uav_id,
        })
        from coverage_env.env.time_model import heading_from_path, path_time
        # The junction "previous valid segment -> this action's first segment" is
        # settled here, once, and charged to THIS step. A cleared flag (takeoff or a
        # resupply stop) means no continuous-flight junction is billed.
        prev_dir = None
        if self._heading_valid.get(result.uav_id, False):
            prev_dir = heading_from_path(self._mission_trajectories.get(result.uav_id, []))
        self._heading_valid[result.uav_id] = True
        trajectory = self._mission_trajectories.setdefault(result.uav_id, [])
        if result.total_path:
            if trajectory and trajectory[-1] == result.total_path[0]:
                trajectory.extend(result.total_path[1:])
            else:
                trajectory.extend(result.total_path)
        # Time ledger: base terms are length / cruise speed and are MUTUALLY
        # EXCLUSIVE (work vs non-work), the maneuver term is only the EXTRA time at
        # direction changes, so no segment is ever counted twice.
        _cfg = self.rl_config.get("time", {})
        _params = {k: _cfg[k] for k in ("cruise_speed", "tau_pi", "exponent")
                   if k in _cfg}
        _full = [tuple(pt) for pt in result.total_path]
        _base, _extra = path_time(prev_dir, _full, **_params)
        _v = _params.get("cruise_speed", 8.0)
        # Service time belongs to a SERVICE EVENT (resupply), not to every coverage
        # configuration; charging it per configuration would make a non-zero value
        # into "pay once per visit".
        _service = 0.0
        feedback = {
            "configuration": configuration,
            "path": list(result.total_path),
            "time_work_base": float(result.coverage_distance) / _v,
            "time_nonwork_base": max(0.0, _base - float(result.coverage_distance) / _v),
            "time_maneuver_extra": float(_extra),
            "time_service": _service,
            "time_total": float(_base + _extra + _service),
            "distance": result.distance,
            "energy": result.energy,
            "payload": result.payload,
            "transfer_distance": result.transfer_distance,
            "coverage_distance": result.coverage_distance,
            "coverage": result.covered_area,
            "entry_boundary_id": configuration.entry_boundary_id,
            "exit_boundary_id": configuration.exit_boundary_id,
            "entry_point": result.entry_boundary_point,
            "exit_point": result.exit_boundary_point,
            "entry_boundary_point": result.entry_boundary_point,
            "coverage_start_point": result.coverage_start_point,
            "exit_boundary_point": result.exit_boundary_point,
            "entry_transit_path": list(result.entry_transit_path),
            "exit_transit_path": list(result.exit_transit_path),
            "coverage_path": list(result.coverage_path),
            "coverage_strokes": list(result.coverage_strokes),
            "scan_transit_paths": [list(path) for path in result.scan_transit_paths],
            "growth_edge_id": result.growth_edge_id,
            "max_reachable_area": result.max_reachable_area,
            "remaining_points": list(result.remaining_points),
            "total_path": list(result.total_path),
            "state": self.get_state(),
            "new_state": self.get_state(),
        }
        self._action_estimate_cache.clear()
        self.episode_logger.log_transition(
            before_state,
            configuration,
            {
                key: feedback[key]
                for key in (
                    "path", "distance", "energy", "coverage",
                    "entry_point", "exit_point",
                    "entry_boundary_id", "exit_boundary_id",
                    "entry_boundary_point", "coverage_start_point",
                    "exit_boundary_point", "entry_transit_path",
                    "coverage_path", "exit_transit_path", "total_path",
                    "coverage_strokes", "scan_transit_paths",
                    "growth_edge_id", "max_reachable_area", "remaining_points",
                )
            },
            feedback["state"],
        )
        return feedback

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """Apply one action and return ``(next_state, reward, done, info)``.

        Expected planning/resource failures become penalized transitions with
        no physical state change. Legacy callers can continue to use
        ``step_configuration`` for its original feedback dictionary.
        """
        from coverage_env.planner.configuration import CoverageConfiguration
        from coverage_env.rl.action_space import CoverageAction
        from coverage_env.rl.reward import calculate_reward

        before_state = self.get_state()
        try:
            if isinstance(action, CoverageAction):
                workload = self.estimate_workload(action, before_state)
                workload = self._refine_workload_for_execution(action, workload)
                if workload <= 0.0:
                    return self._failed_step(
                        action, before_state,
                        "No positive feasible workload for selected action.",
                    )
                configuration = CoverageConfiguration(
                    action.field_id, action.entry_boundary_id,
                    action.exit_boundary_id, workload,
                )
            elif isinstance(action, CoverageConfiguration):
                configuration = action
                spatial = CoverageAction(
                    action.field_id, action.entry_boundary_id, action.exit_boundary_id
                )
                maximum = self.estimate_workload(spatial, before_state)
                maximum = self._refine_workload_for_execution(spatial, maximum)
                if action.workload_ratio > maximum + 1e-8:
                    return self._failed_step(
                        action, before_state,
                        "Requested workload exceeds the environment-estimated "
                        "feasible workload.",
                    )
            else:
                return self._failed_step(
                    action, before_state,
                    "action must be a CoverageAction or CoverageConfiguration.",
                )
        except (KeyError, ValueError, TypeError, RuntimeError, IndexError) as exc:
            return self._failed_step(action, before_state, str(exc))

        snapshot = self._capture_physical_state()
        try:
            feedback = self.step_configuration(configuration)
        except (KeyError, ValueError, TypeError, RuntimeError, IndexError) as exc:
            self._restore_physical_state(snapshot)
            return self._failed_step(action, before_state, str(exc))

        result = self._execution_records[-1]
        info = {
            "success": True,
            "selected_action": {
                "field_id": configuration.field_id,
                "entry_boundary_id": configuration.entry_boundary_id,
                "exit_boundary_id": configuration.exit_boundary_id,
            },
            "configuration": configuration,
            "selected_field": configuration.field_id,
            "entry_boundary": configuration.entry_boundary_id,
            "exit_boundary": configuration.exit_boundary_id,
            "workload": configuration.workload_ratio,
            "estimated_workload": configuration.workload_ratio,
            "path_length": feedback["distance"],
            # Time ledger, mirrored from the execution feedback so step consumers do
            # not have to reach into the result object.
            "time_work_base": feedback["time_work_base"],
            "time_nonwork_base": feedback["time_nonwork_base"],
            "time_maneuver_extra": feedback["time_maneuver_extra"],
            "time_service": feedback["time_service"],
            "time_total": feedback["time_total"],
            "time_resupply_or_return": 0.0,
            # Diagnostic split. ``path_length`` covers only the executed coverage
            # segment; the resupply and final-return legs are appended below so that
            # sum(path_length) closes to summary["total_distance"]. The reward and
            # total_distance are unchanged -- both already include those legs.
            "coverage_only_distance": float(feedback.get("coverage_distance", 0.0)),
            "transit_distance": float(feedback.get("transfer_distance", 0.0)),
            "resupply_or_return_distance": 0.0,
            "energy": feedback["energy"],
            "coverage_gain": feedback["coverage"],
            "exit_point": feedback["exit_point"],
            "path": feedback["path"],
            "result": result,
        }
        reward = calculate_reward(
            feedback, self.rl_config, total_area=self.total_area
        )
        self._growth_area_cache.clear()
        self._consecutive_failures = 0
        info["auto_resupply"] = False
        info["resupply_count"] = self._resupply_count

        if self._all_fields_complete():
            done = True
            info["termination_reason"] = "mission_complete"
            try:
                return_feedback = self.finalize_episode()
            except ValueError as exc:
                return_feedback = {"completed": False, "reason": str(exc)}
            if return_feedback is not None:
                _leg = float(return_feedback.get("distance", 0.0))
                for key in ("time_total", "time_maneuver_extra", "time_nonwork_base"):
                    return_feedback.setdefault(key, 0.0)
                reward += calculate_reward(
                    return_feedback, self.rl_config, total_area=self.total_area
                )
                info["return_to_depot"] = return_feedback
                info["path_length"] = float(info.get("path_length", 0.0)) + _leg
                info["time_total"] += return_feedback["time_total"]
                info["time_nonwork_base"] += return_feedback["time_nonwork_base"]
                info["time_maneuver_extra"] += return_feedback["time_maneuver_extra"]
                info["time_resupply_or_return"] += return_feedback["time_total"]
                if self.uavs:
                    self._heading_valid[self.uavs[0].uav_id] = False
                info["resupply_or_return_distance"] += _leg
                info["final_return_distance"] = _leg
                if return_feedback.get("path"):
                    info["path"] = list(info.get("path", [])) + list(return_feedback["path"])
        else:
            actions = self.get_action_candidates()
            feasible = bool(actions) and any(self.get_action_mask(actions))
            if feasible:
                done = False
            else:
                resupply = self._attempt_auto_resupply()
                if resupply is not None:
                    _rf = resupply["return_feedback"]
                    _svc = float(self.rl_config.get("time", {}).get("service_time", 0.0))
                    _rf["time_service"] = _svc
                    _rf["time_total"] += _svc
                    reward += calculate_reward(
                        _rf, self.rl_config, total_area=self.total_area,
                    )
                    _base_length = float(info.get("path_length", 0.0))
                    _base_path = list(info.get("path", []))
                    _leg = float(resupply["return_feedback"].get("distance", 0.0))
                    _v = float(self.rl_config.get("time", {}).get("cruise_speed", 8.0))
                    _leg_time = _leg / _v
                    info.update(resupply["info"])
                    info["time_total"] = float(info.get("time_total", 0.0)) + _rf["time_total"]
                    info["time_nonwork_base"] = float(info.get("time_nonwork_base", 0.0)) + _rf["time_nonwork_base"]
                    info["time_maneuver_extra"] = float(info.get("time_maneuver_extra", 0.0)) + _rf["time_maneuver_extra"]
                    info["time_resupply_or_return"] = _rf["time_total"]
                    if _rf.get("time_service"):
                        info["time_service"] = _rf["time_service"]
                    info["path_length"] = _base_length + _leg
                    info["resupply_or_return_distance"] = _leg
                    info["resupply_distance"] = _leg
                    info["path"] = _base_path + list(resupply["return_feedback"].get("path", []))
                    done = False
                else:
                    done = True
                    info["termination_reason"] = self._termination_reason

        next_state = self.get_state()
        self._time_totals["work"] += float(info.get("time_work_base", 0.0))
        self._time_totals["nonwork"] += float(info.get("time_nonwork_base", 0.0))
        self._time_totals["maneuver"] += float(info.get("time_maneuver_extra", 0.0))
        self._time_totals["service"] += float(info.get("time_service", 0.0))
        self._time_totals["resupply_or_return"] += float(info.get("time_resupply_or_return", 0.0))
        self._time_totals["total"] += float(info.get("time_total", 0.0))
        return EnvironmentStep(next_state, reward, done, info)

    def _capture_physical_state(self) -> dict:
        """Snapshot mutable mission state so failed execution can roll back."""
        return {
            "uavs": [deepcopy(uav.__dict__) for uav in self.uavs],
            "covered": [
                [(point, point.covered) for point in field.grid_points]
                for field in self.fields
            ],
            "trajectories": deepcopy(self._mission_trajectories),
            "records": len(self._execution_records),
            "logs": len(self.episode_logger._transitions),
        }

    def _restore_physical_state(self, snapshot: dict) -> None:
        """Restore UAV, coverage, trajectory, and log state after a failed step."""
        for uav, saved in zip(self.uavs, snapshot["uavs"]):
            uav.__dict__.clear()
            uav.__dict__.update(saved)
        for field_points in snapshot["covered"]:
            for point, was_covered in field_points:
                point.covered = was_covered
        self._mission_trajectories = snapshot["trajectories"]
        del self._execution_records[snapshot["records"]:]
        del self.episode_logger._transitions[snapshot["logs"]:]
        self._action_estimate_cache.clear()

    def _failed_step(self, action, before_state: dict, reason: str):
        """Return a penalized, non-mutating failure transition for PPO."""
        self._consecutive_failures += 1
        threshold = max(1, int(self.rl_config.get("failure_termination_threshold", 5)))
        done = self._consecutive_failures >= threshold
        next_state = self.get_state()
        failure_penalty = float(self.rl_config.get("failure_penalty", -1.0))
        info = {
            "success": False,
            "failed": True,
            "failure_reason": reason or "Selected action could not be executed.",
            "action": action,
            "selected_field": getattr(action, "field_id", None),
            "entry_boundary": getattr(action, "entry_boundary_id", None),
            "exit_boundary": getattr(action, "exit_boundary_id", None),
            "workload": getattr(action, "workload_ratio", 0.0),
            "path_length": 0.0,
            "energy": 0.0,
            "payload": 0.0,
            "coverage_gain": 0.0,
            "exit_point": None,
            "failure_count": self._consecutive_failures,
        }
        self.episode_logger.log_transition(
            before_state, action, info, next_state
        )
        return EnvironmentStep(next_state, failure_penalty, done, info)

    def get_action_candidates(self, workload_options=None):
        """Generate currently possible high-level actions."""
        from coverage_env.rl.action_space import ActionGenerator

        return ActionGenerator().generate(self)

    def estimate_workload(self, action, state=None) -> float:
        """Estimate the largest positive workload feasible for a spatial action.

        The estimate is constrained by the entry-reachable uncovered component,
        the current UAV energy, and a return-to-depot reserve. Workload is an
        execution-layer quantity and is therefore absent from ``CoverageAction``.
        """
        from coverage_env.rl.action_space import CoverageAction

        if not isinstance(action, CoverageAction):
            raise TypeError("estimate_workload expects a CoverageAction.")
        try:
            field = self.get_field(action.field_id)
        except KeyError:
            return 0.0
        if not self.uavs:
            return 0.0
        uav = self.uavs[0]
        cache_key = (
            action.field_id, action.entry_boundary_id, action.exit_boundary_id,
            tuple(uav.current_position or self.depot.get_position()),
            float(uav.remaining_energy),
            float(uav.remaining_payload),
            tuple(point.id for point in field.get_uncovered_points()),
        )
        if cache_key in self._action_estimate_cache:
            return self._action_estimate_cache[cache_key]
        ratio = self._estimate_workload_uncached(action, field, uav)
        self._action_estimate_cache[cache_key] = ratio
        return ratio

    def _estimate_workload_uncached(self, action, field, uav) -> float:
        """Calculate the max-workload estimate for one cache miss."""
        from math import hypot, sqrt

        if not field.get_uncovered_points():
            return 0.0
        try:
            entry = field.get_boundary(action.entry_boundary_id)
            exit_boundary = field.get_boundary(action.exit_boundary_id)
        except (KeyError, TypeError):
            return 0.0
        if entry.length <= 0.0 or exit_boundary.length <= 0.0:
            return 0.0
        from coverage_env.planner.configuration_executor import ConfigurationExecutor
        from coverage_env.planner.boundary_region_generator import BoundaryRegionGenerator
        from coverage_env.planner.growth_edge import GrowthEdgeSelector

        current = uav.current_position or self.depot.get_position()
        entry_point = ConfigurationExecutor._project_to_boundary(entry, current)
        try:
            selection_key = (field.field_id, entry.id, exit_boundary.id)
            growth_selection = self._growth_selection_cache.get(selection_key)
            if growth_selection is None:
                growth_selection = GrowthEdgeSelector().select(
                    field, entry, exit_boundary
                )
                self._growth_selection_cache[selection_key] = growth_selection
            coverage_key = (
                field.field_id,
                tuple(point.id for point in field.get_uncovered_points()),
                growth_selection.growth_edge.id,
            )
            reachable_area = self._growth_area_cache.get(coverage_key)
            if reachable_area is None:
                reachable_area = BoundaryRegionGenerator().max_reachable_area(
                    field, growth_selection.growth_edge
                )
                self._growth_area_cache[coverage_key] = reachable_area
        except (ValueError, KeyError):
            return 0.0
        remaining_area = sum(point.point_area for point in field.get_uncovered_points())
        maximum_area = min(reachable_area, remaining_area)
        if uav.payload_per_area > 0.0:
            maximum_area = min(
                maximum_area, uav.remaining_payload / uav.payload_per_area
            )
        if maximum_area <= 0.0:
            return 0.0

        entry_distance = hypot(entry_point[0] - current[0], entry_point[1] - current[1])
        return_distance = hypot(
            exit_boundary.center[0] - self.depot.position[0],
            exit_boundary.center[1] - self.depot.position[1],
        )
        if uav.remaining_energy <= 0.0:
            return 0.0

        width = max(float(uav.coverage_width), 1e-9)
        alpha, beta = max(0.0, uav.alpha), max(0.0, uav.beta)
        safety_factor = max(
            1.0, float(self.rl_config.get("fast_estimate_safety_factor", 1.25))
        )
        def estimated_energy(area):
            sweep = 1.25 * (area / width + 2.0 * sqrt(max(area, 0.0)))
            estimated_route = entry_distance + sweep + return_distance
            return safety_factor * (alpha * estimated_route + beta * area)

        if estimated_energy(maximum_area) <= uav.remaining_energy:
            feasible_area = maximum_area
        else:
            low, high = 0.0, maximum_area
            for _ in range(40):
                mid = (low + high) / 2.0
                if estimated_energy(mid) <= uav.remaining_energy:
                    low = mid
                else:
                    high = mid
            feasible_area = low
        point_area = min(
            (p.point_area for p in field.get_uncovered_points()), default=0.0
        )
        if feasible_area + 1e-9 < point_area:
            return 0.0
        maximum_ratio = min(1.0, feasible_area / remaining_area)
        # Keep action-mask evaluation inexpensive. The selected action is
        # checked against the exact planner and resource use in ``step``.
        return maximum_ratio

    def _refine_workload_for_execution(self, action, upper_bound: float) -> float:
        """Check estimated workload against the real executor without mutation."""
        if upper_bound <= 0.0 or not self.uavs:
            return 0.0
        from copy import copy, deepcopy
        from math import hypot
        from types import SimpleNamespace
        from coverage_env.planner.configuration import CoverageConfiguration
        from coverage_env.planner.configuration_executor import ConfigurationExecutor

        field_copy = _copy_field_for_preview(self.get_field(action.field_id), copy, deepcopy)
        uav_copy = deepcopy(self.uavs[0])
        depot = self.depot
        sandbox = SimpleNamespace(
            fields=[field_copy], uavs=[uav_copy], depot=depot,
            get_field=lambda _field_id: field_copy,
        )
        executor = ConfigurationExecutor(
            float(self.rl_config.get("grid_resolution", 5.0))
        )
        low, high = 0.0, min(1.0, float(upper_bound))

        def is_feasible(ratio: float) -> bool:
            """Run one isolated real-plan preview and reserve return energy."""
            trial_field = deepcopy(field_copy)
            trial_uav = deepcopy(uav_copy)
            sandbox.fields = [trial_field]
            sandbox.uavs = [trial_uav]
            sandbox.get_field = lambda _field_id, f=trial_field: f
            try:
                result = executor.execute(
                    CoverageConfiguration(
                        action.field_id, action.entry_boundary_id,
                        action.exit_boundary_id, ratio,
                    ),
                    sandbox,
                )
            except (ValueError, KeyError, RuntimeError, IndexError):
                return False
            return_distance = hypot(
                result.exit_boundary_point[0] - depot.position[0],
                result.exit_boundary_point[1] - depot.position[1],
            )
            total_required = result.energy + trial_uav.alpha * return_distance
            if result.payload > self.uavs[0].remaining_payload + 1e-8:
                return False
            return total_required <= self.uavs[0].remaining_energy + 1e-8

        # The rough energy model is usually conservative enough. Check its
        # upper bound once and avoid fourteen redundant full geometry plans.
        if is_feasible(high):
            return high

        best = 0.0
        # Only refine when the estimated upper bound exceeds the real budget. Each
        # probe is a full geometry plan, so this loop is the dominant cost of a step
        # (measured: 12 previews per step on the clustered instances). Five bisection
        # steps resolve the workload ratio to 1/32, which is far below the payload and
        # energy granularity that decides feasibility.
        for _ in range(5):
            ratio = (low + high) / 2.0
            if ratio <= 0.0:
                break
            if is_feasible(ratio):
                best = ratio
                low = ratio
            else:
                high = ratio
        return best

    def get_action_mask(self, actions=None):
        """Return a feasibility mask aligned with candidate action order."""
        from coverage_env.rl.action_mask import get_action_mask

        return get_action_mask(self.get_action_candidates() if actions is None else actions, self)

    def _has_feasible_action_at_full_resources(self) -> bool:
        """Probe whether any action is feasible at the depot with full resources."""
        if not self.uavs:
            return False
        uav = self.uavs[0]
        saved = (uav.current_position, uav.remaining_energy, uav.remaining_payload)
        try:
            uav.current_position = self.depot.get_position()
            uav.remaining_energy = uav.max_energy
            uav.remaining_payload = uav.max_payload
            actions = self.get_action_candidates()
            if not actions:
                return False
            return any(self.get_action_mask(actions))
        finally:
            uav.current_position, uav.remaining_energy, uav.remaining_payload = saved

    def _attempt_auto_resupply(self) -> dict | None:
        """Auto-return + refill when resource depletion blocks every action.

        Returns ``{"return_feedback": ..., "info": ...}`` when resupply completed,
        or ``None`` after setting ``self._termination_reason`` for termination.
        """
        if not self.rl_config.get("allow_auto_resupply", True):
            self._termination_reason = "auto_resupply_disabled"
            return None
        if not self._has_feasible_action_at_full_resources():
            self._termination_reason = "no_feasible_action_at_full_resources"
            return None
        uav = self.uavs[0]
        from math import hypot

        start = uav.current_position or self.depot.get_position()
        depot = self.depot.get_position()
        return_distance = hypot(start[0] - depot[0], start[1] - depot[1])
        return_energy = uav.alpha * return_distance
        if uav.remaining_energy < return_energy:
            self._termination_reason = "return_infeasible"
            return None

        before_energy = uav.remaining_energy
        before_payload = uav.remaining_payload
        return_feedback = self.return_to_depot()
        self._heading_valid[uav.uav_id] = False
        replenished = uav.replenish()
        self._resupply_count += 1
        info = {
            "auto_resupply": True,
            "resupply_count": self._resupply_count,
            "return_distance": float(return_feedback["distance"]),
            "return_energy": float(return_feedback["energy"]),
            "remaining_energy_before_resupply": before_energy,
            "remaining_payload_before_resupply": before_payload,
            "remaining_energy_after_resupply": uav.remaining_energy,
            "remaining_payload_after_resupply": uav.remaining_payload,
            "resupply_reason": "no_feasible_coverage_action",
            "energy_recharged": float(replenished["energy_recharged"]),
            "payload_resupplied": float(replenished["payload_resupplied"]),
        }
        return {"return_feedback": return_feedback, "info": info}

    def _all_fields_complete(self) -> bool:
        """Return True when every field has no uncovered grid points."""
        return all(not field.get_uncovered_points() for field in self.fields)

    def return_to_depot(self) -> dict:
        """Fly every UAV straight back to the depot and report the cost.

        Unlike the reserves used during workload estimation, this terminal leg
        actually moves the UAVs and charges their batteries.
        """
        depot_position = self.depot.get_position()
        per_uav = {}
        total_distance = 0.0
        total_energy = 0.0
        total_base = total_extra = 0.0
        params = {k: v for k, v in self.rl_config.get("time", {}).items()
                  if k in ("cruise_speed", "tau_pi", "exponent")}
        for uav in self.uavs:
            start = uav.current_position or depot_position
            leg = [start, depot_position]
            prev = (heading_from_path(self._mission_trajectories.get(uav.uav_id, []))
                    if self._heading_valid.get(uav.uav_id, False) else None)
            base, extra = path_time(prev, leg, **params)
            feedback = uav.execute_path([depot_position], area=0.0)
            trajectory = self._mission_trajectories.setdefault(uav.uav_id, [])
            if start != depot_position:
                leg = [start, depot_position]
                if trajectory and trajectory[-1] == leg[0]:
                    trajectory.extend(leg[1:])
                else:
                    trajectory.extend(leg)
            self._heading_valid[uav.uav_id] = False
            total_base += base
            total_extra += extra
            per_uav[uav.uav_id] = {
                "path": leg,
                "time_nonwork_base": base,
                "time_maneuver_extra": extra,
                "time_total": base + extra,
                "distance": feedback["path_length"],
                "energy": feedback["energy_consumption"],
                "remaining_energy": feedback["remaining_energy"],
                "final_position": feedback["final_position"],
            }
            total_distance += feedback["path_length"]
            total_energy += feedback["energy_consumption"]
        _leg_path = []
        if per_uav:
            first = next(iter(per_uav.values()))
            _leg_path = first["path"]
        return {
            "time_total": total_base + total_extra,
            "time_nonwork_base": total_base,
            "time_maneuver_extra": total_extra,
            "path": _leg_path if _leg_path else [depot_position],
            "distance": total_distance,
            "energy": total_energy,
            "coverage": 0.0,
            "per_uav": per_uav,
            "final_position": self.uavs[0].current_position if self.uavs else depot_position,
        }

    def finalize_episode(self) -> dict | None:
        """Fly the UAVs back to the depot once the mission is complete.

        This is the shared "real return" hook for both the RL ``step`` path and
        the heuristic baseline episode loop. Returns the return-leg feedback
        dict, or ``None`` when the mission is not complete or
        ``rl.return_to_depot`` is disabled.
        """
        if not self.rl_config.get("return_to_depot", True):
            return None
        if not self._all_fields_complete():
            return None
        return self.return_to_depot()

    def is_done(self) -> bool:
        """Return true when all fields are covered or no feasible action remains."""
        if self._all_fields_complete():
            return True
        actions = self.get_action_candidates()
        if not actions:
            return True
        from coverage_env.rl.action_mask import get_action_mask

        return not any(get_action_mask(actions, self))

    def get_summary(self) -> dict:
        """Return mission summary; includes the cumulative flight-time ledger."""
        """Return cumulative mission costs, coverage, and UAV trajectories."""
        total_field_area = sum(field.area for field in self.fields)
        covered_area = sum(field.calculate_coverage_area() for field in self.fields)
        completed_fields = []
        for field in self.fields:
            remaining = field.get_uncovered_points()
            if field.grid_points and not remaining:
                completed_fields.append(field.field_id)
        _t = getattr(self, "_time_totals", None) or {}
        _parts = (float(_t.get("work", 0.0)) + float(_t.get("nonwork", 0.0))
                  + float(_t.get("maneuver", 0.0)) + float(_t.get("service", 0.0)))
        return {
            # Cumulative flight time. ``total_time`` is the four-term sum, and
            # ``time_closure_error`` exposes any accounting drift instead of hiding it.
            "total_time": float(_t.get("total", 0.0)),
            "time_work_base": float(_t.get("work", 0.0)),
            "time_nonwork_base": float(_t.get("nonwork", 0.0)),
            "time_maneuver_extra": float(_t.get("maneuver", 0.0)),
            "time_service": float(_t.get("service", 0.0)),
            "time_resupply_or_return": float(_t.get("resupply_or_return", 0.0)),
            "time_closure_error": _parts - float(_t.get("total", 0.0)),
            "total_distance": sum(uav.total_distance for uav in self.uavs),
            "total_energy": sum(uav.total_energy_consumed for uav in self.uavs),
            "coverage_ratio": min(1.0, covered_area / total_field_area)
            if total_field_area > 0 else 0.0,
            "covered_area": covered_area,
            "remaining_energy": {
                uav.uav_id: uav.remaining_energy for uav in self.uavs
            },
            "total_payload": sum(uav.total_payload_consumed for uav in self.uavs),
            "remaining_payload": {
                uav.uav_id: uav.remaining_payload for uav in self.uavs
            },
            "resupply_count": self._resupply_count,
            "total_energy_recharged": sum(
                uav.total_energy_recharged for uav in self.uavs
            ),
            "total_payload_resupplied": sum(
                uav.total_payload_resupplied for uav in self.uavs
            ),
            "completed_fields": completed_fields,
            "uav_trajectory": {
                uav_id: list(path)
                for uav_id, path in self._mission_trajectories.items()
            },
            "execution_count": len(self._execution_records),
            "returned_to_depot": bool(
                self.uavs
                and all(
                    uav.current_position == self.depot.get_position()
                    for uav in self.uavs
                )
            ),
        }

    @property
    def execution_records(self) -> tuple[dict, ...]:
        """Return immutable access to the executed configuration records."""
        return tuple(self._execution_records)


# Backward-compatible name used by the Phase 3/4 examples and callers.
Environment = CoverageEnvironment


def _copy_field_for_preview(field, shallow_copy, deep_copy):
    """Copy mutable coverage points while sharing immutable field geometry."""
    field_copy = shallow_copy(field)
    field_copy.grid_points = deep_copy(field.grid_points)
    return field_copy
