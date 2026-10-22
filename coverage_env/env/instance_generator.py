"""Random, reproducible multi-field test-instance generation."""

from __future__ import annotations

import random

from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split as split_geometry
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
        elif mode == "tessellated":
            fields, block_parcels = self._sample_tessellated_fields(
                rng, num_fields, layout, (min_x, min_y, max_x, max_y),
                workspace_transformer, min_area, max_area,
            )
        elif mode == "clustered":
            fields, block_parcels = self._sample_clustered_fields(
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

    def _sample_tessellated_fields(
        self, rng, num_fields, layout, bounds, transformer, min_area, max_area,
    ) -> list[Field]:
        """Subdivide ONE land block into parcels along farm tracks, like real farmland.

        Real parcels are convex, usually four-sided (rectangles or trapezoids) and
        arranged in bands along the working direction, all sharing boundaries; they
        are not a jigsaw of triangles. See the field-boundary datasets built from
        aerial imagery (AI4Boundaries, ESSD 2023).

        So the block is cut into a few **rows** first, then each row is cut across
        into parcels. Cut angles carry a small jitter so the result is trapezoids
        rather than a perfect grid, and neighbouring parcels share a boundary that is
        inset by ``parcel_gap_m / 2`` to become a farm track.

        Only ``num_fields`` of the resulting parcels are worked: a farm does not treat
        every parcel on the same day, and the unworked ones become the real gaps that
        give the visiting order structure.
        """
        min_x, min_y, max_x, max_y = bounds
        gap = max(0.0, float(layout.get("parcel_gap_m", 1.5)))
        mean_area = 0.5 * (min_area + max_area)
        oversample = max(1.0, float(layout.get("tessellated_oversample", 2.0)))
        pool_size = max(num_fields, int(round(num_fields * oversample)))

        for _attempt in range(200):
            # Block sized so the whole pool of parcels fills it.
            # The 8 m track is cut OUT of each parcel, so the block must be bigger by
            # the inset loss or every parcel would end up under min_area after
            # insetting (~25% loss for a 1 ha parcel with an 8 m track).
            shrink = 1.0 - min(0.45, gap / max((mean_area ** 0.5), 1e-6))
            side = (pool_size * mean_area / max(shrink * shrink, 0.35)) ** 0.5
            aspect = rng.uniform(0.8, 1.4)
            width, height = side * aspect, side / max(aspect, 1e-6)
            centre = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
            block = Polygon([
                (centre[0] - width / 2, centre[1] - height / 2),
                (centre[0] + width / 2, centre[1] - height / 2),
                (centre[0] + width / 2, centre[1] + height / 2),
                (centre[0] - width / 2, centre[1] + height / 2),
            ])
            dominant = rng.uniform(0.0, pi)
            # Rows: long bands along the working direction.
            rows = max(2, min(4, int(round(pool_size / 5.0))))
            row_pieces = self._band_cuts(rng, block, dominant + pi / 2.0, rows)
            if row_pieces is None:
                continue
            parcels = []
            for piece in row_pieces:
                share = max(1, int(round(pool_size * piece.area / block.area)))
                strips = self._band_cuts(rng, piece, dominant, share)
                if strips is None:
                    parcels = None
                    break
                parcels.extend(strips)
            if not parcels:
                continue
            if any(not (min_area * 0.9 <= piece.area <= max_area * 1.3) for piece in parcels):
                continue
            # Three- and five-sided parcels are legitimate (a track cutting a corner
            # leaves a triangle beside a pentagon); only reject degenerate shapes.
            if any(not 3 <= len(list(piece.exterior.coords)) - 1 <= 6 for piece in parcels):
                continue

            parcels = self._corner_cut(rng, parcels, min_area, max_area, 0.14)

            # Inset so shared boundaries become farm tracks, and drop individual
            # pieces that no longer make a valid parcel. Rejecting the whole
            # tessellation for one bad piece made generation fail ~7% of the time
            # (an 8 m track removes ~25% of a 1 ha parcel, so borderline pieces are
            # common); dropping just those and checking the count is enough.
            objects = []
            for piece in parcels:
                inset = piece.buffer(-gap / 2.0) if gap > 0.0 else piece
                if inset.is_empty or inset.geom_type != "Polygon" or not inset.is_valid:
                    continue
                if not (min_area <= inset.area <= max_area):
                    continue
                if len(list(inset.exterior.coords)) - 1 < 3:
                    continue
                # 4*pi*area/perimeter^2 is 1 for a circle, ~0.8 for a square and below
                # 0.05 only for needle-shaped pieces that collapse when projected.
                if 4.0 * pi * inset.area / max(inset.length, 1e-9) ** 2 < 0.05:
                    continue
                objects.append(inset)
            if len(objects) < num_fields:
                continue

            if len(objects) > num_fields:
                keep = sorted(rng.sample(range(len(objects)), num_fields))
                objects = [objects[index] for index in keep]

            # Building the Field projects the outline to lat/lon and back, which can
            # still fail for a marginal shape no local check catches; retry the whole
            # tessellation rather than propagate.
            try:
                fields = []
                for order, piece in enumerate(objects, start=1):
                    points = list(piece.exterior.coords)[:-1]
                    fields.append(self._field_with_area(
                        f"field-{order}", points, transformer, piece.area,
                        min_area, max_area,
                    ))
            except (ValueError, RuntimeError):
                continue
            block_parcels = [list(piece.exterior.coords)[:-1] for piece in objects]
            return fields, block_parcels

        raise RuntimeError(
            "Unable to tessellate the field block; loosen field_generator.field_size."
        )

    def _corner_cut(self, rng, parcels, min_area, max_area, probability=0.14):
        """Slice a corner off a minority of parcels, leaving a triangle and a pentagon.

        Real boundaries are mostly four-sided, but a road or track cutting a corner
        leaves a triangle next to a pentagon. Cutting a convex quad through two
        opposite sides can only ever give quads, so this cuts near a corner on purpose:
        the triangle sits at the corner and the remainder gains a fifth side.
        """
        result = []
        for piece in parcels:
            if rng.random() > probability:
                result.append(piece)
                continue
            corners = list(piece.exterior.coords)[:-1]
            if len(corners) < 4:
                result.append(piece)
                continue
            index = rng.randrange(len(corners))
            corner = corners[index]
            before = corners[index - 1]
            after = corners[(index + 1) % len(corners)]
            # The corner triangle at fractions (t, t) along the two edges has area
            # t^2 times the area of the full corner triangle, so the cut position is
            # available in closed form.
            full = abs(
                (before[0] - corner[0]) * (after[1] - corner[1])
                - (after[0] - corner[0]) * (before[1] - corner[1])
            ) / 2.0
            if full <= 0.0:
                result.append(piece)
                continue
            target = rng.uniform(max(min_area, 0.35 * piece.area),
                                 min(max_area, 0.5 * piece.area))
            ratio = max(0.05, min(0.95, (target / full) ** 0.5))
            point_a = (corner[0] + ratio * (before[0] - corner[0]),
                       corner[1] + ratio * (before[1] - corner[1]))
            point_b = (corner[0] + ratio * (after[0] - corner[0]),
                       corner[1] + ratio * (after[1] - corner[1]))
            triangle = Polygon([corner, point_a, point_b])
            if triangle.area < min_area or triangle.area > max_area:
                result.append(piece)
                continue
            remainder = piece.difference(triangle)
            candidate = []
            for part in getattr(remainder, "geoms", [remainder]):
                if part.geom_type == "Polygon" and part.area > 0.0:
                    candidate.append(part)
            if len(candidate) != 1:
                result.append(piece)
                continue
            pentagon = candidate[0]
            if not (min_area <= pentagon.area <= max_area):
                result.append(piece)
                continue
            result.extend([triangle, pentagon])
        return result

    def _band_cuts(self, rng, polygon, angle, count, tilt_probability=0.12):
        """Cut ``polygon`` into ``count`` bands across ``angle`` with jittered cuts.

        Each cut is a straight line close to perpendicular to the band direction, so
        the pieces stay convex and four-sided instead of degenerating into triangles.
        """
        if count <= 1:
            return [polygon]
        bounds_i = polygon.bounds
        extent = max(bounds_i[2] - bounds_i[0], bounds_i[3] - bounds_i[1])
        # Most cuts run across the band (square corners, four-sided parcels), but real
        # boundaries are not all square: a track cutting a corner leaves a triangle
        # next to a pentagon. A minority of cuts is therefore tilted.
        direction = (cos(angle + pi / 2.0), sin(angle + pi / 2.0))
        # Cut positions: evenly spaced with jitter, expressed as fractions along the
        # polygon's projection on the cut normal.
        normal = (-direction[1], direction[0])
        corners = list(polygon.exterior.coords)
        projections = [normal[0] * x + normal[1] * y for x, y in corners]
        low, high = min(projections), max(projections)
        span = high - low
        if span <= 0.0:
            return None
        fractions = sorted(
            (index + rng.uniform(-0.22, 0.22)) / count for index in range(1, count)
        )
        pieces = [polygon]
        for fraction in fractions:
            if not 0.05 < fraction < 0.95:
                continue
            offset = low + span * fraction
            origin = (normal[0] * offset, normal[1] * offset)
            reach = 4.0 * extent
            if rng.random() < tilt_probability:
                skew = rng.choice((-1.0, 1.0)) * rng.uniform(0.12, 0.30)
                cut_direction = (
                    cos(angle + pi / 2.0 + skew), sin(angle + pi / 2.0 + skew)
                )
            else:
                cut_direction = direction
            cut = LineString([
                (origin[0] - cut_direction[0] * reach, origin[1] - cut_direction[1] * reach),
                (origin[0] + cut_direction[0] * reach, origin[1] + cut_direction[1] * reach),
            ])
            updated = []
            for piece in pieces:
                for part in split_geometry(piece, cut).geoms:
                    updated.extend(
                        part.geoms if isinstance(part, MultiPolygon) else [part]
                    )
            pieces = [piece for piece in updated if not piece.is_empty and piece.area > 0.0]
            if len(pieces) != len(updated):
                return None
        return pieces if len(pieces) == count else None

    def _sample_clustered_fields(
        self, rng, num_fields, layout, bounds, transformer, min_area, max_area,
    ) -> list[Field]:
        """Parcels grouped into a few tight clusters separated by real tracks.

        Real farmland is asymmetric: parcels inside one block sit a few metres apart
        across a farm track, while separate blocks are tens to hundreds of metres
        apart. That asymmetry is what makes the visiting order worth optimising.

        The uniform ``contiguous`` layout did not produce it. Measured: configured
        ``gap_m`` was 15 m but the realised boundary clearance was 41-45 m, and the
        parcels covered only 10-16% of their block -- close to a sparse lattice, where
        every hop is the same length and "nearest neighbour" is already near-optimal.

        Placement here bisects on the *real polygon-to-polygon distance* instead of
        using circumscribed circles, so the requested clearance is the achieved one.
        """
        min_x, min_y, max_x, max_y = bounds
        shape = self.config["field_generator"]["shape"]
        inner_gap = max(0.0, float(layout.get("cluster_inner_gap_m", 4.0)))
        outer_gap = max(inner_gap, float(layout.get("cluster_gap_m", 90.0)))
        attempts = max(4, int(layout.get("growth_attempts", 60)))
        cluster_count = min(
            int(layout.get("cluster_max", 4)),
            max(int(layout.get("cluster_min", 2)), int(ceil(num_fields / 5.0))),
        )
        cluster_count = max(1, min(cluster_count, num_fields))
        sizes = [num_fields // cluster_count] * cluster_count
        for index in range(num_fields - sum(sizes)):
            sizes[index % cluster_count] += 1

        def clearance(points_a, points_b) -> float:
            if len(points_a) < 3 or len(points_b) < 3:
                return float("inf")
            return Polygon(points_a).distance(Polygon(points_b))

        def bisect_offset(anchor, local, angle, target, centre):
            """Offset along ``angle`` from ``centre`` giving exactly ``target`` gap."""
            cosine, sine = cos(angle), sin(angle)

            def gap_at(distance):
                moved = [
                    (centre[0] + distance * cosine + x, centre[1] + distance * sine + y)
                    for x, y in local
                ]
                return clearance(moved, anchor)

            low, high = 0.0, 1.0
            while gap_at(high) < target and high < 1e6:
                high *= 2.0
                low = high / 2.0
            for _ in range(48):
                mid = 0.5 * (low + high)
                if gap_at(mid) < target:
                    low = mid
                else:
                    high = mid
            return high

        def grow(count: int):
            """Grow ``count`` parcels, each placed clear of *every* earlier parcel.

            An earlier version bisected the distance against the anchor parcel only, so
            the third parcel could clear its anchor while landing on top of another one.
            Measured: 5/150 parcel pairs intersected at n=5, with up to 8,399 m2 of
            overlap -- close to a whole parcel. The clearance must therefore be checked
            against all placed parcels, and since the minimum over them is not monotone
            along a ray, this scans the ray and bisects only the final bracket.
            """
            placed: list[tuple[list[tuple[float, float]], tuple[float, float]]] = []
            for _ in range(count):
                target_area = rng.uniform(min_area, max_area)
                polygon = self._irregular_polygon_xy(rng, (0.0, 0.0), target_area, shape)
                centre, radius = self._polygon_center_and_radius(polygon)
                local = [(x - centre[0], y - centre[1]) for x, y in polygon]
                if not placed:
                    placed.append((local, (0.0, 0.0)))
                    continue

                def min_gap(origin, angle, distance):
                    """Smallest clearance from the candidate to every placed parcel."""
                    cosine, sine = cos(angle), sin(angle)
                    moved = [
                        (origin[0] + distance * cosine + x,
                         origin[1] + distance * sine + y) for x, y in local
                    ]
                    worst = float("inf")
                    for other, position in placed:
                        absolute = [(position[0] + x, position[1] + y) for x, y in other]
                        worst = min(worst, clearance(moved, absolute))
                        if worst < inner_gap:
                            return worst
                    return worst

                position = None
                ordered = sorted(placed, key=lambda entry: entry[1])
                for _attempt in range(max(4, attempts)):
                    anchor_points, anchor_position = ordered[rng.randrange(len(ordered))]
                    angle = rng.uniform(0.0, 2.0 * pi)
                    # Scan outward from the anchor until clear of everyone, then
                    # bisect that last bracket so the parcel stays as close as it can.
                    previous = 0.0
                    distance = 0.0
                    limit = 6.0 * radius + 400.0
                    while distance <= limit:
                        if min_gap(anchor_position, angle, distance) >= inner_gap:
                            break
                        previous = distance
                        distance += max(2.0, radius / 8.0)
                    if distance > limit:
                        continue
                    low, high = previous, distance
                    for _step in range(30):
                        mid = 0.5 * (low + high)
                        if min_gap(anchor_position, angle, mid) >= inner_gap:
                            high = mid
                        else:
                            low = mid
                    position = (
                        anchor_position[0] + high * cos(angle),
                        anchor_position[1] + high * sin(angle),
                    )
                    break
                if position is None:
                    raise RuntimeError(
                        "Unable to grow a clustered field block; grow the workspace or "
                        "reduce field_generator.field_size."
                    )
                placed.append((local, position))
            return placed

        clusters = [grow(count) for count in sizes]

        # Park clusters so their real clearance is ``outer_gap``.
        cluster_positions = [(0.0, 0.0)]
        for index in range(1, len(clusters)):
            anchor_index = rng.randrange(index)
            anchor_centre = cluster_positions[anchor_index]
            best, best_distance = None, 0.0
            for _attempt in range(24):
                angle = rng.uniform(0.0, 2.0 * pi)
                # Grow the offset until every parcel pair clears ``outer_gap``.
                def all_clear(distance):
                    centre_now = (anchor_centre[0] + distance * cos(angle),
                                  anchor_centre[1] + distance * sin(angle))
                    # Every earlier cluster must be respected, not only the anchor:
                    # checking just the anchor lets a later cluster park right next to
                    # an earlier one, which merges the blocks (measured: max nearest
                    # clearance collapsed to 4 m, i.e. one blob).
                    for earlier in range(index):
                        other_centre = cluster_positions[earlier]
                        for _local, position in clusters[index]:
                            moved = [(centre_now[0] + position[0] + x,
                                      centre_now[1] + position[1] + y) for x, y in _local]
                            for other_local, other_position in clusters[earlier]:
                                anchor_points = [
                                    (other_centre[0] + other_position[0] + x,
                                     other_centre[1] + other_position[1] + y)
                                    for x, y in other_local
                                ]
                                if clearance(moved, anchor_points) < outer_gap:
                                    return False
                    return True

                low, high = 0.0, 1.0
                while not all_clear(high) and high < 1e6:
                    high *= 2.0
                if not all_clear(high):
                    continue
                for _step in range(40):
                    mid = 0.5 * (low + high)
                    if all_clear(mid):
                        high = mid
                    else:
                        low = mid
                best = (anchor_centre[0] + high * cos(angle),
                        anchor_centre[1] + high * sin(angle))
                best_distance = high
                break
            if best is None:
                raise RuntimeError(
                    "Unable to separate the field clusters; reduce cluster_gap_m."
                )
            cluster_positions.append(best)

        parcels: list[tuple[float, list[tuple[float, float]]]] = []
        for centre, cluster in zip(cluster_positions, clusters):
            for local, position in cluster:
                parcels.append((
                    rng.uniform(min_area, max_area),
                    [(centre[0] + position[0] + x, centre[1] + position[1] + y)
                     for x, y in local],
                ))
        rng.shuffle(parcels)
        parcels = parcels[:num_fields]

        mean_x = sum(sum(x for x, _ in pts) / len(pts) for _a, pts in parcels) / len(parcels)
        mean_y = sum(sum(y for _, y in pts) / len(pts) for _a, pts in parcels) / len(parcels)
        shift_x = (min_x + max_x) / 2.0 - mean_x
        shift_y = (min_y + max_y) / 2.0 - mean_y

        fields = []
        for order, (target_area, points) in enumerate(parcels, start=1):
            moved = [(shift_x + x, shift_y + y) for x, y in points]
            fields.append(self._field_with_area(
                f"field-{order}", moved, transformer, target_area, min_area, max_area,
            ))
        block_parcels = [
            [(shift_x + x, shift_y + y) for x, y in points] for _a, points in parcels
        ]
        return fields, block_parcels

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
        # Both packed layouts place the depot as farm headquarters a short standoff
        # beyond the block. `clustered` used to fall through to the generic random
        # branch, which dropped it anywhere in the enlarged workspace -- measured
        # 1300-1700 m from the parcels instead of the configured 50-400 m.
        if mode in ("contiguous", "clustered", "tessellated") and block_parcels:
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
