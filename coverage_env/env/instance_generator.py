"""Random, reproducible multi-field test-instance generation."""

from __future__ import annotations

import random
from math import ceil, cos, hypot, pi, sin, sqrt

from coverage_env.env.depot import Depot
from coverage_env.env.environment import CoverageEnvironment
from coverage_env.env.field import Field
from coverage_env.env.uav import UAV
from coverage_env.geometry.coordinate import CoordinateTransformer
from coverage_env.geometry.discretization import discretize_field


class InstanceGenerator:
    """Generate valid irregular fields, a depot, and configured UAVs."""

    def __init__(self, config: dict) -> None:
        """Store the experiment config and validate required sections."""
        self.config = config
        # Only used by environment.num_fields_strategy == "cycle".
        self._field_cycle_index = 0
        for section in ("environment", "field_generator", "depot", "uav", "coverage"):
            if section not in config:
                raise ValueError(f"Missing configuration section: {section}")

    def generate(
        self,
        *,
        num_fields: int | None = None,
        seed: int | None = None,
        randomize_each_reset: bool | None = None,
    ) -> CoverageEnvironment:
        """Generate a fully initialized ``CoverageEnvironment``.

        ``num_fields`` and ``seed`` override the default config for experiment
        cases while all geometry and vehicle parameters still come from config.

        ``randomize_each_reset`` controls cross-episode generalization: when it
        is enabled the returned environment resamples fields (count and
        geometry), the depot, and UAV capability parameters on every
        ``reset()`` using its own ``random.Random``. It defaults to ``True``
        when ``num_fields`` is omitted (training over the configured range) and
        to ``False`` for an explicit fixed count (single-instance experiments).
        """
        environment_config = self.config["environment"]
        rng = random.Random(
            environment_config.get("random_seed", 0) if seed is None else seed
        )
        if randomize_each_reset is None:
            randomize_each_reset = num_fields is None
        count = self.sample_num_fields(rng, num_fields)
        fields, depot, uavs, resolution = self.sample_components(
            rng, count, randomize_uav=bool(randomize_each_reset)
        )

        environment = CoverageEnvironment(
            fields,
            depot,
            uavs,
            rl_config=self.config.get("rl", {}),
            config=self.config,
            rng=rng,
            randomize_each_reset=bool(randomize_each_reset),
        )
        for field in environment.fields:
            discretize_field(field, resolution)
        environment.reset()
        return environment

    def sample_num_fields(self, rng, num_fields: int | None = None) -> int:
        """Return a validated field count, drawing one when ``num_fields`` is None."""
        count_config = self.config["environment"]["num_fields"]
        minimum = int(count_config["min"])
        maximum = int(count_config["max"])
        if num_fields is None:
            strategy = str(
                self.config["environment"].get("num_fields_strategy", "uniform")
            ).lower()
            if strategy == "cycle":
                # Walk the configured range in order instead of drawing uniformly.
                # Each update then covers a predictable set of field counts, so the
                # logged mean stops depending on which counts the sampler happened to
                # draw -- that drift is 4.03 reward points across 5..20 fields, i.e.
                # larger than a whole long run's improvement.
                span = maximum - minimum + 1
                num_fields = minimum + (self._field_cycle_index % span)
                self._field_cycle_index += 1
            elif strategy == "uniform":
                num_fields = rng.randint(minimum, maximum)
            else:
                raise ValueError(
                    f"Unsupported environment.num_fields_strategy: {strategy!r}"
                )
        if not minimum <= int(num_fields) <= maximum:
            raise ValueError(
                f"num_fields={num_fields} is outside configured range "
                f"[{minimum}, {maximum}]."
            )
        return int(num_fields)

    def sample_components(
        self, rng, num_fields: int, *, randomize_uav: bool = False,
        attempts: int = 8,
    ):
        """Sample ``(fields, depot, uavs, resolution)`` without building an env.

        Both initial generation and per-episode resampling run through this
        pipeline, so field geometry and vehicle randomization stay in one place.

        Field placement is rejection sampling (``_sample_center`` tries a bounded
        number of random positions), so a crowded workspace can occasionally
        fail even for a field count that is usually placeable. Cross-episode
        generalization resamples on *every* ``reset()``, which turns that rare
        bad draw into an eventual crash -- so retry the whole draw a few times
        before giving up. Retries only consume more RNG; they do not change the
        sampling distribution.
        """
        last_error: RuntimeError | None = None
        for _ in range(max(1, int(attempts))):
            try:
                return self._sample_components_once(
                    rng, num_fields, randomize_uav=randomize_uav
                )
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(
            f"Field placement failed after {attempts} attempts for "
            f"num_fields={num_fields}; the workspace is too crowded for this "
            f"field count."
        ) from last_error

    def _sample_components_once(
        self, rng, num_fields: int, *, randomize_uav: bool = False
    ):
        """Draw one candidate instance; raises if field placement fails."""
        environment_config = self.config["environment"]
        workspace = environment_config["workspace"]
        origin_lat = (float(workspace["min_lat"]) + float(workspace["max_lat"])) / 2
        origin_lon = (float(workspace["min_lon"]) + float(workspace["max_lon"])) / 2
        workspace_transformer = CoordinateTransformer(origin_lat, origin_lon)
        min_x, min_y, max_x, max_y = self._workspace_xy_bounds(
            workspace, workspace_transformer
        )

        field_configs = self.config["field_generator"]
        size_config = field_configs["field_size"]
        min_area = float(size_config["min_area"])
        max_area = float(size_config["max_area"])
        if min_area <= 0.0 or max_area < min_area:
            raise ValueError("field_generator.field_size area limits are invalid.")

        layout = dict(field_configs.get("layout") or {})
        mode = str(layout.get("mode", "scattered"))
        block_parcels = None
        if mode == "scattered":
            fields = self._sample_scattered_fields(
                rng, num_fields, (min_x, min_y, max_x, max_y),
                workspace_transformer, min_area, max_area,
            )
        elif mode == "contiguous":
            fields, block_parcels = self._sample_contiguous_fields(
                rng, num_fields, layout, (min_x, min_y, max_x, max_y),
                workspace_transformer, min_area, max_area,
            )
        else:
            raise ValueError(f"Unsupported field_generator.layout.mode: {mode!r}.")

        depot_position = self._depot_position(
            rng, fields, workspace, layout=layout, block_parcels=block_parcels,
            workspace_bounds=(min_x, min_y, max_x, max_y),
            transformer=workspace_transformer,
        )
        depot = Depot("depot-1", depot_position)
        uavs = self._build_uavs(rng, randomize_uav=randomize_uav)

        resolution = float(field_configs["resolution"]["grid_resolution"])
        return fields, depot, uavs, resolution

    def _sample_scattered_fields(
        self, rng, num_fields, bounds, transformer, min_area, max_area,
    ) -> list[Field]:
        """Place fields uniformly at random over the whole workspace.

        Kept so pre-existing baselines stay reproducible; real farmland looks like
        :meth:`_sample_contiguous_fields` instead. Draw order (area, then centre)
        matches the historical implementation so the same seed yields the same
        instance.
        """
        min_x, min_y, max_x, max_y = bounds
        field_configs = self.config["field_generator"]
        shape = field_configs["shape"]
        irregularity = float(shape.get("irregularity", 0.2))
        radial_max = float(shape.get("radial_max", 1.0 + irregularity))
        radius_bound = sqrt(max_area) * radial_max * 1.25
        spacing = float(field_configs.get("min_field_spacing", 0.0))
        centers_and_radii: list[tuple[float, float, float]] = []
        fields = []
        for field_index in range(num_fields):
            target_area = rng.uniform(min_area, max_area)
            center = self._sample_center(
                rng, min_x, min_y, max_x, max_y,
                radius_bound, spacing, centers_and_radii,
            )
            polygon_xy = self._irregular_polygon_xy(rng, center, target_area, shape)
            field = self._field_with_area(
                f"field-{field_index + 1}", polygon_xy,
                transformer, target_area, min_area, max_area,
            )
            fields.append(field)
            centers_and_radii.append((center[0], center[1], radius_bound))
        return fields

    def _sample_contiguous_fields(
        self, rng, num_fields, layout, bounds, transformer, min_area, max_area,
    ) -> list[Field]:
        """Grow a contiguous parcel pool, then randomly select ``num_fields``.

        Real farmland is a block of neighbouring parcels separated by farm tracks,
        not fields scattered across kilometres. A larger pool is grown by placing
        each new parcel tangent to an already-placed one (``gap_m`` apart), and the
        episode then uses a random subset. That keeps the adjacency while still
        giving every episode a different instance, and the subset leaves genuine
        gaps between the chosen parcels -- a farm does not treat every parcel on
        the same day.
        """
        min_x, min_y, max_x, max_y = bounds
        shape = self.config["field_generator"]["shape"]
        gap = max(0.0, float(layout.get("gap_m", 15.0)))
        oversample = max(1.0, float(layout.get("pool_oversample", 1.5)))
        attempts = max(1, int(layout.get("growth_attempts", 60)))
        pool_size = max(num_fields, int(ceil(num_fields * oversample)))

        # Each pool entry is (target_area, local_polygon, position, radius).
        pool: list[tuple[float, list[tuple[float, float]],
                         tuple[float, float], float]] = []
        for _ in range(pool_size):
            target_area = rng.uniform(min_area, max_area)
            polygon = self._irregular_polygon_xy(rng, (0.0, 0.0), target_area, shape)
            center, radius = self._polygon_center_and_radius(polygon)
            local = [(x - center[0], y - center[1]) for x, y in polygon]

            if not pool:
                position = (0.0, 0.0)
            else:
                position = None
                for _attempt in range(attempts):
                    _anchor, _polygon, anchor_center, anchor_radius = pool[
                        rng.randrange(len(pool))
                    ]
                    angle = rng.uniform(0.0, 2.0 * pi)
                    distance = anchor_radius + radius + gap
                    candidate = (
                        anchor_center[0] + distance * cos(angle),
                        anchor_center[1] + distance * sin(angle),
                    )
                    if all(
                        hypot(candidate[0] - other_center[0],
                              candidate[1] - other_center[1])
                        >= radius + other_radius + gap
                        for _area, _poly, other_center, other_radius in pool
                    ):
                        position = candidate
                        break
                if position is None:
                    raise RuntimeError(
                        "Unable to grow a contiguous field block; reduce "
                        "field_generator.field_size or raise growth_attempts."
                    )
            pool.append((target_area, local, position, radius))

        chosen = sorted(rng.sample(range(len(pool)), num_fields))
        selected = [pool[index] for index in chosen]
        # Centre the selected block inside the workspace so the depot offset and
        # the position normalisers behave the same way across instances.
        mean_x = sum(entry[2][0] for entry in selected) / len(selected)
        mean_y = sum(entry[2][1] for entry in selected) / len(selected)
        shift_x = (min_x + max_x) / 2.0 - mean_x
        shift_y = (min_y + max_y) / 2.0 - mean_y

        fields = []
        for order, (target_area, local, position, _radius) in enumerate(selected, start=1):
            moved = [
                (position[0] + shift_x + x, position[1] + shift_y + y)
                for x, y in local
            ]
            fields.append(self._field_with_area(
                f"field-{order}", moved, transformer, target_area,
                min_area, max_area,
            ))
        # Per-parcel outlines in the *shared* local frame, used to place the depot.
        # They cannot be read back off the fields here: each Field still carries its
        # own per-parcel transformer until the environment installs the shared one,
        # so ``xy_polygon`` would be centred on each parcel instead of on the block.
        block_parcels = [
            [(position[0] + shift_x + x, position[1] + shift_y + y) for x, y in local]
            for _area, local, position, _radius in selected
        ]
        return fields, block_parcels

    @staticmethod
    def _polygon_center_and_radius(points) -> tuple[tuple[float, float], float]:
        """Return a polygon's vertex-mean centre and farthest-vertex distance."""
        center_x = sum(point[0] for point in points) / len(points)
        center_y = sum(point[1] for point in points) / len(points)
        radius = max(hypot(x - center_x, y - center_y) for x, y in points)
        return (center_x, center_y), radius

    def _build_uavs(self, rng, *, randomize_uav: bool = False) -> list[UAV]:
        """Build ``num_uav`` UAVs, sampling capability ranges when requested."""
        uav_config = self.config["uav"]
        energy_config = uav_config["energy"]
        ranges = dict(uav_config.get("ranges") or {}) if randomize_uav else {}
        uavs = []
        for index in range(int(uav_config["num_uav"])):
            max_payload = self._sample_range(
                rng, ranges.get("max_payload"), uav_config.get("max_payload")
            )
            uavs.append(
                UAV(
                    f"UAV-{index + 1}",
                    max_energy=float(self._sample_range(
                        rng, ranges.get("max_energy"), uav_config["max_energy"]
                    )),
                    speed=float(self._sample_range(
                        rng, ranges.get("speed"), uav_config["speed"]
                    )),
                    coverage_width=float(self._sample_range(
                        rng, ranges.get("coverage_width"),
                        uav_config["coverage_width"],
                    )),
                    sensor_width=float(uav_config["sensor_width"]),
                    energy_consumption_parameters={
                        "alpha": float(self._sample_range(
                            rng, ranges.get("alpha"), energy_config["alpha"]
                        )),
                        "beta": float(self._sample_range(
                            rng, ranges.get("beta"), energy_config["beta"]
                        )),
                    },
                    max_payload=None if max_payload is None else float(max_payload),
                    payload_per_area=float(self._sample_range(
                        rng, ranges.get("payload_per_area"),
                        uav_config.get("payload_per_area", 0.0),
                    )),
                )
            )
        return uavs

    @staticmethod
    def _sample_range(rng, limits, fallback):
        """Return a uniform draw from ``{"min", "max"}`` or the ``fallback`` value.

        Ranges are only consulted when they are configured, so fixed-instance
        generation consumes no RNG state for vehicle parameters.
        """
        if not limits:
            return fallback
        low = float(limits["min"])
        high = float(limits["max"])
        if high < low:
            raise ValueError(f"UAV range min {low} exceeds max {high}.")
        return rng.uniform(low, high)

    @staticmethod
    def _workspace_xy_bounds(workspace: dict, transformer: CoordinateTransformer):
        """Project workspace corners and return a metric bounding box."""
        corners = [
            transformer.to_xy(float(latitude), float(longitude))
            for latitude in (workspace["min_lat"], workspace["max_lat"])
            for longitude in (workspace["min_lon"], workspace["max_lon"])
        ]
        return (
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        )

    @staticmethod
    def _sample_center(
        rng, min_x, min_y, max_x, max_y, radius, spacing, placed
    ) -> tuple[float, float]:
        """Sample a center with enough spacing to avoid field overlap."""
        for _ in range(5000):
            x = rng.uniform(min_x + radius, max_x - radius)
            y = rng.uniform(min_y + radius, max_y - radius)
            if all(
                sqrt((x - old_x) ** 2 + (y - old_y) ** 2)
                >= radius + old_radius + spacing
                for old_x, old_y, old_radius in placed
            ):
                return x, y
        raise RuntimeError(
            "Unable to place non-overlapping fields in configured workspace; "
            "increase workspace bounds or reduce field size/count."
        )

    @staticmethod
    def _irregular_polygon_xy(rng, center, target_area, shape) -> list[tuple[float, float]]:
        """Create a simple radial polygon and scale it to target area."""
        vertex_count = rng.randint(
            int(shape["min_vertices"]), int(shape["max_vertices"])
        )
        irregularity = float(shape.get("irregularity", 0.2))
        radial_min = float(shape.get("radial_min", 1.0 - irregularity))
        radial_max = float(shape.get("radial_max", 1.0 + irregularity))
        if not 0.0 < radial_min <= radial_max:
            raise ValueError("shape radial_min/radial_max must be positive and ordered.")
        angles = sorted(
            (2.0 * pi * index / vertex_count + rng.uniform(-0.10, 0.10))
            % (2.0 * pi)
            for index in range(vertex_count)
        )
        radii = [rng.uniform(radial_min, radial_max) for _ in angles]
        unscaled = [(r * cos(angle), r * sin(angle)) for r, angle in zip(radii, angles)]
        local_area = InstanceGenerator._polygon_area(unscaled)
        scale = sqrt(target_area / local_area)
        return [
            (center[0] + x * scale, center[1] + y * scale)
            for x, y in unscaled
        ]

    @staticmethod
    def _polygon_area(points) -> float:
        """Calculate the absolute shoelace area of a ring."""
        return abs(sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )) / 2.0

    @staticmethod
    def _field_with_area(
        field_id, polygon_xy, transformer, target_area, min_area, max_area
    ) -> Field:
        """Project a polygon to WGS84 and adjust for projection-area drift."""
        points = polygon_xy
        for _ in range(5):
            coordinates = [transformer.to_latlon(x, y) for x, y in points]
            field = Field(field_id, coordinates)
            if min_area <= field.area <= max_area:
                return field
            factor = sqrt(target_area / field.area)
            center_x = sum(point[0] for point in points) / len(points)
            center_y = sum(point[1] for point in points) / len(points)
            points = [
                (center_x + (x - center_x) * factor,
                 center_y + (y - center_y) * factor)
                for x, y in points
            ]
        # Raise rather than silently return an instance outside its constraints.
        raise ValueError(
            f"Generated field {field_id} area {field.area:.2f} m^2 is outside "
            f"[{min_area:.2f}, {max_area:.2f}] m^2."
        )

    def _depot_position(
        self, rng, fields, workspace, *, layout=None, block_parcels=None,
        workspace_bounds=None, transformer=None,
    ):
        """Create a configured or random depot in the shared local frame."""
        depot_config = self.config["depot"]
        position = depot_config.get("position")
        if position is not None:
            return float(position[0]), float(position[1])
        depot_type = depot_config.get("type", "random")
        if depot_type != "random":
            raise ValueError(f"Unsupported depot.type: {depot_type!r}.")
        layout = layout or {}
        mode = str(layout.get("mode", "scattered"))
        if mode == "contiguous" and block_parcels:
            # Farm headquarters. Rules, in order:
            #   1. pick any direction around the block (not one of four corners);
            #   2. sit STANDOFF metres beyond the block's actual extent *in that
            #      direction*, measured as the support function of the parcel
            #      vertices -- using the bounding box instead let the depot land
            #      anywhere from touching a parcel to 435 m away, because the
            #      cluster is far smaller than its box;
            #   3. stay inside the workspace so the position normalisers hold.
            limits = layout.get("depot_offset_m", {"min": 50.0, "max": 400.0})
            if isinstance(limits, dict):
                low = float(limits.get("min", 50.0))
                high = float(limits.get("max", 400.0))
            else:
                low = high = float(limits)
            low, high = min(low, high), max(low, high)
            flat = [point for parcel in block_parcels for point in parcel]
            centre_x = sum(point[0] for point in flat) / len(flat)
            centre_y = sum(point[1] for point in flat) / len(flat)
            angle = rng.uniform(0.0, 2.0 * pi)
            direction_x, direction_y = cos(angle), sin(angle)
            standoff = rng.uniform(low, high) if high > low else low
            # Bisect along the ray for the point whose distance to the NEAREST
            # parcel equals the sampled standoff. Using the support function
            # instead only guaranteed the standoff in the sampled direction, so the
            # nearest parcel could still be touching the depot (measured 0 m) or
            # several hundred metres away (measured 679 m).
            from shapely.geometry import Point
            from shapely.geometry import Polygon as ShapelyPolygon

            parcels = [ShapelyPolygon(parcel) for parcel in block_parcels]
            farthest = max(
                hypot(point[0] - centre_x, point[1] - centre_y) for point in flat
            )

            def nearest_distance(distance: float) -> float:
                probe = Point(
                    centre_x + direction_x * distance,
                    centre_y + direction_y * distance,
                )
                return min(parcel.distance(probe) for parcel in parcels)

            lower, upper = 0.0, farthest + high + 200.0
            for _ in range(48):
                middle = (lower + upper) / 2.0
                if nearest_distance(middle) < standoff:
                    lower = middle
                else:
                    upper = middle
            distance = upper
            position_x = centre_x + direction_x * distance
            position_y = centre_y + direction_y * distance
            if workspace_bounds is not None:
                min_x, min_y, max_x, max_y = workspace_bounds
                position_x = min(max(position_x, min_x + 10.0), max_x - 10.0)
                position_y = min(max(position_y, min_y + 10.0), max_y - 10.0)
            # The block geometry above lives in the *workspace* frame, but the
            # environment installs ``fields[0].coordinate_transformer`` as the
            # shared frame (whose origin is field-1's own centroid), so the point
            # must be carried across via WGS84. Skipping this shifted the depot by
            # the workspace-centre-to-field-1 offset and made its true standoff
            # range from 0 to ~680 m instead of the sampled value.
            if transformer is not None and fields:
                latitude, longitude = transformer.to_latlon(position_x, position_y)
                return fields[0].coordinate_transformer.to_xy(latitude, longitude)
            return (position_x, position_y)
        latitude = rng.uniform(float(workspace["min_lat"]), float(workspace["max_lat"]))
        longitude = rng.uniform(float(workspace["min_lon"]), float(workspace["max_lon"]))
        return fields[0].coordinate_transformer.to_xy(latitude, longitude)


def generate_instance(
    config: dict, *, num_fields: int | None = None, seed: int | None = None
) -> CoverageEnvironment:
    """Convenience function for one randomized Environment instance."""
    return InstanceGenerator(config).generate(num_fields=num_fields, seed=seed)
