"""Regression tests for the turn/distance ledger.

The motivating incident: an audit read a dict with getattr, silently obtained empty
paths, and reported "turns never change". These tests make that failure loud.
"""
import unittest

from coverage_env.rl.evaluation.turn_ledger import (
    build_ledger, heading_profile, require_record,
)


def _record(**over):
    base = {"coverage_path": [(0, 0), (10, 0), (10, 5)],
            "coverage_strokes": [[(0, 0), (10, 0)], [(0, 5), (10, 5)]],
            "total_path": [(0, 0), (10, 0), (10, 5)],
            "entry_transit_path": [(0, 0)], "exit_transit_path": [(10, 5)],
            "distance": 15.0, "field_id": "f1", "coverage": 50.0,
            "entry_boundary_id": "B0", "exit_boundary_id": "B1"}
    base.update(over)
    return base


class RecordReadingTests(unittest.TestCase):
    def test_missing_key_raises(self):
        rec = _record()
        del rec["coverage_path"]
        with self.assertRaises(KeyError):
            require_record(rec, covered_area=50.0)

    def test_empty_geometry_with_positive_area_raises(self):
        for key in ("coverage_path", "coverage_strokes"):
            with self.assertRaises(ValueError):
                require_record(_record(**{key: []}), covered_area=50.0)

    def test_empty_geometry_is_allowed_for_zero_area(self):
        require_record(_record(coverage_path=[], coverage_strokes=[]), covered_area=0.0)

    def test_non_mapping_raises(self):
        class Obj:
            coverage_path = []
        with self.assertRaises(TypeError):
            require_record(Obj(), covered_area=0.0)


class HeadingTests(unittest.TestCase):
    def test_known_reversal_is_non_zero(self):
        # a sweep reversal: fly +x, then fly -x => one 180 degree event
        path = [(0, 0), (10, 0), (10, 1), (0, 1)]
        total, turns, stops = heading_profile(path)
        self.assertGreater(total, 170.0)
        self.assertEqual(len(turns), 2)          # two 90 degree corners
        self.assertEqual(stops, [])

    def test_straight_path_has_no_turns(self):
        total, turns, _ = heading_profile([(0, 0), (5, 0), (10, 0)])
        self.assertEqual((total, len(turns)), (0.0, 0))


class LedgerClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        from coverage_env.env.instance_generator import InstanceGenerator
        from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice

        config = load_config(DEFAULT_CONFIG_PATH)
        env = InstanceGenerator(config).generate(
            num_fields=5, seed=30000, randomize_each_reset=False)
        env.reset()
        done = False
        while not done:
            action = _nearest_field_choice(env, None)
            if action is None:
                break
            _s, _r, done, _i = env.step(action)
        cls.env = env
        cls.ledger = build_ledger(env)

    def test_distance_closes_to_reported_total(self):
        mission = self.ledger["mission"]
        self.assertAlmostEqual(mission["distance_closure_error"], 0.0, places=3,
                               msg="完整轨迹距离未闭合到 summary total_distance")

    def test_mission_heading_is_non_zero(self):
        self.assertGreater(self.ledger["mission"]["full_heading_deg"], 0.0,
                           "完整轨迹航向变化不得为零")

    def test_coverage_heading_is_non_zero(self):
        total = sum(v["coverage_heading_deg"] for v in self.ledger["visits"])
        self.assertGreater(total, 0.0)

    def test_per_field_visits_are_consistent(self):
        fields = self.ledger["fields"]
        self.assertEqual(sum(len(f["visits"]) for f in fields.values()),
                         self.ledger["mission"]["n_visits"])


if __name__ == "__main__":
    unittest.main()
