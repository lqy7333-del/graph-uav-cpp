"""Random, reproducible multi-field test-instance generation."""

from __future__ import annotations

import random
from math import cos, pi, sin, sqrt

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
            num_fields = rng.randint(minimum, maximum)
        if not minimum <= int(num_fields) <= maximum:
            raise ValueError(
                f"num_fields={num_fields} is outside configured range "
                f"[{minimum}, {maximum}]."
            )
        return int(num_fields)

    def sample_components(
        self, rng, num_fields: int, *, randomize_uav: bool = False
    ):
        """Sample ``(fields, depot, uavs, resolution)`` without building an env.

        Both initial generation and per-episode resampling run through this
        pipeline, so field geometry and vehicle randomization stay in one place.
        """
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

        centers_and_radii: list[tuple[float, float, float]] = []
        fields = []
        spacing = float(field_configs.get("min_field_spacing", 0.0))
        for field_index in range(num_fields):
            target_area = rng.uniform(min_area, max_area)
            shape = field_configs["shape"]
            irregularity = float(shape.get("irregularity", 0.2))
            radial_max = float(shape.get("radial_max", 1.0 + irregularity))
            radius_bound = sqrt(max_area) * radial_max * 1.25
            center = self._sample_center(
                rng, min_x, min_y, max_x, max_y,
                radius_bound, spacing, centers_and_radii,
            )
            polygon_xy = self._irregular_polygon_xy(
                rng, center, target_area, shape
            )
            field = self._field_with_area(
                f"field-{field_index + 1}", polygon_xy,
                workspace_transformer, target_area,
                min_area, max_area,
            )
            fields.append(field)
            centers_and_radii.append((center[0], center[1], radius_bound))

        depot_position = self._depot_position(
            rng, fields, workspace
        )
        depot = Depot("depot-1", depot_position)
        uavs = self._build_uavs(rng, randomize_uav=randomize_uav)

        resolution = float(field_configs["resolution"]["grid_resolution"])
        return fields, depot, uavs, resolution

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
                    payload_per_area=float(uav_config.get("payload_per_area", 0.0)),
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

    def _depot_position(self, rng, fields, workspace):
        """Create a configured or random depot in the shared local frame."""
        depot_config = self.config["depot"]
        position = depot_config.get("position")
        if position is not None:
            return float(position[0]), float(position[1])
        depot_type = depot_config.get("type", "random")
        if depot_type != "random":
            raise ValueError(f"Unsupported depot.type: {depot_type!r}.")
        latitude = rng.uniform(float(workspace["min_lat"]), float(workspace["max_lat"]))
        longitude = rng.uniform(float(workspace["min_lon"]), float(workspace["max_lon"]))
        return fields[0].coordinate_transformer.to_xy(latitude, longitude)


def generate_instance(
    config: dict, *, num_fields: int | None = None, seed: int | None = None
) -> CoverageEnvironment:
    """Convenience function for one randomized Environment instance."""
    return InstanceGenerator(config).generate(num_fields=num_fields, seed=seed)
