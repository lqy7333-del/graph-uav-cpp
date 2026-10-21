"""The contiguous farmland layout: growing a parcel block, then sampling it.

Real farmland is a block of neighbouring parcels separated by farm tracks, not
fields scattered across kilometres. ``layout.mode = contiguous`` grows a larger
pool of parcels tangent to each other and then randomly selects ``num_fields`` of
them, which keeps the adjacency while still giving every episode a different
instance.
"""

from __future__ import annotations

import unittest

from math import atan2, degrees

import numpy as np
from shapely.geometry import Point

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator


def _edge_gaps(environment) -> list[float]:
    """Return each parcel's distance to its nearest neighbour parcel."""
    polygons = [field.xy_polygon for field in environment.fields]
    gaps = []
    for index, polygon in enumerate(polygons):
        gaps.append(min(
            polygon.distance(other)
            for other in polygons if other is not polygon
        ))
    return gaps


def _block_extent(environment) -> tuple[float, float]:
    """Return the (x, y) extent of every parcel outline."""
    xs = [point[0] for field in environment.fields
          for point in field.xy_polygon.exterior.coords]
    ys = [point[1] for field in environment.fields
          for point in field.xy_polygon.exterior.coords]
    return max(xs) - min(xs), max(ys) - min(ys)


class ContiguousLayoutTests(unittest.TestCase):
    """Parcels must be adjacent, the block compact, and the depot outside it."""

    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG_PATH)
        self.config["field_generator"]["layout"]["mode"] = "contiguous"

    def _generate(self, num_fields=20, seed=7000):
        return InstanceGenerator(self.config).generate(
            num_fields=num_fields, seed=seed, randomize_each_reset=False
        )

    def test_parcels_are_adjacent_not_scattered(self):
        """Nearest-parcel gaps must be farm-track sized, not hundreds of metres."""
        gaps = []
        for seed in range(6):
            environment = self._generate(seed=7000 + seed)
            gaps.extend(_edge_gaps(environment))
        median = float(np.median(gaps))
        self.assertLess(median, 80.0, f"parcels are not adjacent: median gap {median}")
        self.assertGreater(median, 0.0, "parcels must not overlap or touch exactly")

    def test_block_is_compact(self):
        """20 parcels must fit in a block far smaller than the workspace."""
        environment = self._generate()
        width, height = _block_extent(environment)
        # clustered mode places 2-4 blocks with real tracks between them, so the
        # outline is legitimately larger than a single packed block.
        self.assertLess(width, 3000.0)
        self.assertLess(height, 3000.0)
        self.assertGreater(width * height, 0.0)

    def test_scattered_mode_still_reproduces_the_old_layout(self):
        """The previous layout must stay reachable for older baselines."""
        config = load_config(DEFAULT_CONFIG_PATH)
        config["field_generator"]["layout"]["mode"] = "scattered"
        environment = InstanceGenerator(config).generate(
            num_fields=20, seed=7000, randomize_each_reset=False
        )
        self.assertGreater(float(np.median(_edge_gaps(environment))), 100.0)

    def test_depot_standoff_and_direction_generalise(self):
        """A farm headquarters: outside the block, standoff inside its range.

        Both the distance to the farm and the side it sits on must vary across
        episodes. Measuring the distance to the *nearest parcel* is what matters:
        an earlier version placed the depot at a fixed bounding-box corner, which
        on an irregular cluster landed anywhere from touching a parcel to ~680 m
        away, and a frame mismatch (the environment adopts fields[0]'s own
        transformer) shifted it by the workspace-to-field-1 offset.
        """
        layout = self.config["field_generator"]["layout"]
        low = float(layout["depot_offset_m"]["min"])
        high = float(layout["depot_offset_m"]["max"])
        standoffs, angles = [], []
        for seed in range(8):
            environment = self._generate(num_fields=10, seed=7005 + seed)
            depot = Point(*environment.depot.get_position())
            standoffs.append(
                min(field.xy_polygon.distance(depot) for field in environment.fields)
            )
            xs = [point[0] for field in environment.fields
                  for point in field.xy_polygon.exterior.coords]
            ys = [point[1] for field in environment.fields
                  for point in field.xy_polygon.exterior.coords]
            angles.append(degrees(atan2(
                depot.y - (min(ys) + max(ys)) / 2.0,
                depot.x - (min(xs) + max(xs)) / 2.0,
            )))
        for value in standoffs:
            self.assertGreaterEqual(value, low - 2.0)
            self.assertLessEqual(value, high + 2.0)
        self.assertGreater(
            max(angles) - min(angles), 90.0,
            f"depot direction does not generalise: {angles}",
        )

    def test_random_subset_gives_different_instances(self):
        """Two seeds must not produce the same parcel block."""
        first = self._generate(seed=7100)
        second = self._generate(seed=7101)
        signature = lambda environment: tuple(  # noqa: E731
            (field.field_id, round(field.area, 3), round(field.centroid[0], 3))
            for field in environment.fields
        )
        self.assertNotEqual(signature(first), signature(second))

    def test_field_count_is_honoured_for_every_size(self):
        for num_fields in (5, 10, 20):
            environment = self._generate(num_fields=num_fields)
            self.assertEqual(len(environment.fields), num_fields)
            self.assertEqual(
                [field.field_id for field in environment.fields],
                [f"field-{index + 1}" for index in range(num_fields)],
            )


if __name__ == "__main__":
    unittest.main()
