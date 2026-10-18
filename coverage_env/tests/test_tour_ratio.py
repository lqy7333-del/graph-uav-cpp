"""The tour_ratio metric: exact on small instances, and sensitive to ordering.

It replaces the exhaustive reference for routine A/B because that one needs 120
environment episodes per instance, which capped the sample at 10 and left a
Kendall-tau standard error of 0.18 -- larger than the effects being measured.
"""

from __future__ import annotations

import unittest
from math import hypot

from coverage_env.rl.evaluation.tour_ratio import (
    _local_search,
    _nearest_neighbour,
    optimal_tour,
    tour_length,
)


class TourRatioMetricTests(unittest.TestCase):
    """The reference tour must be optimal on small instances."""

    def setUp(self):
        # Four parcels at the corners of a square plus one at the centre.
        self.centroids = {
            "a": (0.0, 0.0), "b": (10.0, 0.0), "c": (10.0, 10.0), "d": (0.0, 10.0),
            "e": (5.0, 5.0),
        }
        self.depot = (-1.0, -1.0)
        self.ids = list(self.centroids)

    def test_tour_length_is_a_round_trip(self):
        order = ["a", "b"]
        expected = (
            hypot(-1.0, -1.0)
            + hypot(10.0, 0.0)
            + hypot(-1.0 - 10.0, -1.0)
        )
        self.assertAlmostEqual(
            tour_length(order, self.centroids, self.depot), expected, places=9
        )
        self.assertEqual(tour_length([], self.centroids, self.depot), 0.0)

    def test_exact_search_matches_brute_force(self):
        order, length = optimal_tour(self.ids, self.centroids, self.depot)
        best = min(
            tour_length(list(perm), self.centroids, self.depot)
            for perm in __import__("itertools").permutations(self.ids)
        )
        self.assertAlmostEqual(length, best, places=9)
        self.assertAlmostEqual(
            tour_length(order, self.centroids, self.depot), best, places=9
        )

    def test_local_search_never_increases_length(self):
        start = _nearest_neighbour(self.ids, self.centroids, self.depot)
        before = tour_length(start, self.centroids, self.depot)
        improved, after = _local_search(start, self.centroids, self.depot)
        self.assertLessEqual(after, before + 1e-9)
        self.assertEqual(sorted(improved), sorted(self.ids))

    def test_ratio_is_one_for_the_optimal_order(self):
        order, length = optimal_tour(self.ids, self.centroids, self.depot)
        self.assertAlmostEqual(
            tour_length(order, self.centroids, self.depot) / length, 1.0, places=9
        )
        # A deliberately bad order must score clearly above 1.
        bad = ["a", "c", "b", "d", "e"]
        self.assertGreater(
            tour_length(bad, self.centroids, self.depot) / length, 1.05
        )


if __name__ == "__main__":
    unittest.main()
