"""Visibility-graph connector: golden routes and prepared-geometry equivalence.

The connector is the rollout's CPU hotspot, so it gets micro-optimised in two
steps (prepared geometry, then a per-geometry subgraph cache). Both must leave
the returned routes **exactly** unchanged; this module pins them.
"""

from __future__ import annotations

import unittest

import torch
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.prepared import prep

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.planner.coverage_path import _shortest_path_inside_polygon
from coverage_env.rl.decoder.action_decoder import CoveragePolicy

# A U-shaped (concave) polygon: the two arms are not mutually visible, so the
# connector must route around the notch through both inner corners.
U_SHAPE = Polygon(
    [(0, 0), (10, 0), (10, 10), (7, 10), (7, 3), (3, 3), (3, 10), (0, 10)]
)

# Captured from the pre-optimisation implementation; these routes are the API.
GOLDEN_PATHS = {
    "U-arms": ((1.0, 9.0), (9.0, 9.0), [(1.0, 9.0), (3.0, 3.0), (7.0, 3.0), (9.0, 9.0)]),
    "U-deep": ((0.5, 0.5), (9.5, 9.5), [(0.5, 0.5), (7.0, 3.0), (9.5, 9.5)]),
    "U-same-arm": ((1.0, 2.0), (1.5, 8.0), [(1.0, 2.0), (1.5, 8.0)]),
    "U-inner-corners": ((3.0, 3.0), (7.0, 3.0), [(3.0, 3.0), (7.0, 3.0)]),
}


def probe_points(geometry) -> list[tuple[float, float]]:
    """Return ring corners plus edge midpoints for every polygon component."""
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    points: list[tuple[float, float]] = []
    for part in parts:
        if not hasattr(part, "exterior"):
            continue
        ring = [(float(x), float(y)) for x, y in list(part.exterior.coords)[:-1]]
        points.extend(ring)
        points.extend(
            ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
            for first, second in zip(ring, ring[1:] + ring[:1])
        )
    return points


def collect_connector_inputs(steps: int = 2, seed: int = 42) -> list[tuple]:
    """Run a small deterministic rollout and record every connector call.

    ``boundary_aware_sweep`` imports the connector lazily, so patching the
    module attribute captures all of them.
    """
    import coverage_env.planner.coverage_path as coverage_path

    # Seed before building the policy so the captured trajectory (and therefore
    # which connectors are exercised) is reproducible across runs.
    torch.manual_seed(seed)
    config = load_config(DEFAULT_CONFIG_PATH)
    config["environment"]["num_fields"] = {"min": 20, "max": 20}
    environment = InstanceGenerator(config).generate(num_fields=20, seed=seed)
    actor = CoveragePolicy(environment, config=config)
    actor.eval()

    captured: list[tuple] = []
    original = coverage_path._shortest_path_inside_polygon

    def spy(start, end, polygon, *, pair_cache=None):
        if polygon is not None and start != end:
            captured.append((start, end, polygon))
        return original(start, end, polygon, pair_cache=pair_cache)

    coverage_path._shortest_path_inside_polygon = spy
    try:
        state = environment.reset()
        with torch.no_grad():
            for _ in range(steps):
                output = actor(state, environment=environment)
                distribution = output["action_distribution"]
                if distribution.numel() == 0:
                    state = environment.reset()
                    continue
                action = output["actions"][
                    int(torch.multinomial(distribution, 1).item())
                ]
                state, _reward, done, _info = environment.step(action)
                if done:
                    state = environment.reset()
    finally:
        coverage_path._shortest_path_inside_polygon = original
    return captured


class VisibilityConnectorTests(unittest.TestCase):
    """Prepared geometry and caching must not change a single waypoint."""

    def test_golden_connector_paths_are_unchanged(self):
        for name, (start, end, expected) in GOLDEN_PATHS.items():
            with self.subTest(name):
                path = _shortest_path_inside_polygon(start, end, U_SHAPE)
                self.assertEqual(
                    [(round(x, 6), round(y, 6)) for x, y in path], expected
                )

    def test_connector_endpoints_and_containment(self):
        for name, (start, end, _expected) in GOLDEN_PATHS.items():
            with self.subTest(name):
                path = _shortest_path_inside_polygon(start, end, U_SHAPE)
                self.assertEqual(path[0], start)
                self.assertEqual(path[-1], end)
                self.assertTrue(
                    U_SHAPE.buffer(1e-9).covers(LineString(path)),
                    f"{name}: connector left the region",
                )

    def test_prepared_predicate_matches_scalar_on_synthetic_geometry(self):
        geometries = [
            U_SHAPE,
            U_SHAPE.buffer(1e-7),
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
            Polygon([(0, 0), (6, 0), (6, 6), (4, 6), (4, 2), (2, 2), (2, 6), (0, 6)]),
            MultiPolygon([U_SHAPE, Polygon([(20, 0), (24, 0), (24, 4), (20, 4)])]),
        ]
        pairs = 0
        for geometry in geometries:
            prepared = prep(geometry)
            points = probe_points(geometry)
            for index, first in enumerate(points):
                for second in points[index + 1:]:
                    segment = LineString([first, second])
                    self.assertEqual(
                        bool(prepared.covers(segment)),
                        bool(geometry.covers(segment)),
                        f"prepared predicate disagrees for {first} -> {second}",
                    )
                    pairs += 1
        self.assertGreater(pairs, 200, "predicate sweep too small to be meaningful")

    def test_prepared_predicate_matches_scalar_on_real_regions(self):
        captured = collect_connector_inputs(steps=2)
        self.assertGreater(len(captured), 10, "no connector calls captured")
        checked = 0
        for start, end, geometry in captured[:20]:
            visible = geometry.buffer(1e-7)
            prepared = prep(visible)
            points = [start, end] + probe_points(geometry)
            for index, first in enumerate(points):
                for second in points[index + 1:]:
                    segment = LineString([first, second])
                    self.assertEqual(
                        bool(prepared.covers(segment)),
                        bool(visible.covers(segment)),
                        f"prepared predicate disagrees for {first} -> {second}",
                    )
                    checked += 1
        self.assertGreater(checked, 500)


    def test_pair_cache_matches_uncached_on_real_inputs(self):
        """The per-region memo must not change a single waypoint."""
        captured = collect_connector_inputs(steps=2)
        by_geometry: dict[int, tuple] = {}
        for start, end, geometry in captured:
            by_geometry.setdefault(id(geometry), (geometry, []))[1].append((start, end))

        checked = 0
        reused = 0
        for geometry, pairs in by_geometry.values():
            cache: dict = {}
            sizes = []
            for start, end in pairs:
                expected = _shortest_path_inside_polygon(start, end, geometry)
                actual = _shortest_path_inside_polygon(
                    start, end, geometry, pair_cache=cache
                )
                self.assertEqual(actual, expected, "cached route differs from uncached")
                sizes.append(len(cache))
                checked += 1
            if len(pairs) > 1:
                # Every connector of one geometry sees the same polygon-only
                # pairs, so after the first call the memo must stop growing.
                self.assertEqual(
                    sizes[0], sizes[-1],
                    "memo kept growing: polygon pairs are not being shared",
                )
                self.assertGreater(sizes[0], 0, "memo was never populated")
                reused += 1
        self.assertGreater(checked, 10, "too few connector calls to be meaningful")
        self.assertGreater(reused, 0, "no geometry was queried more than once")

    def test_pair_cache_is_ignored_for_the_two_endpoints(self):
        """Only polygon-vertex pairs are memoised, never start/end pairs."""
        cache: dict = {}
        _shortest_path_inside_polygon((1.0, 9.0), (9.0, 9.0), U_SHAPE, pair_cache=cache)
        self.assertTrue(cache)
        for first, second in cache:
            self.assertGreaterEqual(first, 2, "start/end pair was cached")
            self.assertGreater(second, first)


if __name__ == "__main__":
    import sys

    captured = collect_connector_inputs(steps=3)
    geometries = {}
    for _start, _end, geometry in captured:
        geometries.setdefault(id(geometry), []).append(geometry)
    print(f"captured {len(captured)} connector calls over {len(geometries)} "
          f"distinct region geometries ({len(captured)/max(1,len(geometries)):.1f} per geometry)")
    for name, (start, end, expected) in GOLDEN_PATHS.items():
        actual = _shortest_path_inside_polygon(start, end, U_SHAPE)
        ok = [(round(x, 6), round(y, 6)) for x, y in actual] == expected
        print(f"  {name:16s}: {'MATCH' if ok else 'DIFFERS'}")
    sys.exit(0)
