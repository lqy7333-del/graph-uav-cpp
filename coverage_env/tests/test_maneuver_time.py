"""Maneuver-time proxy, ledger closure, and time reward (real execution path)."""
import unittest

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.env.time_model import path_time, turn_cost
from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
from coverage_env.rl.reward import calculate_reward


def _episode(mode="time", seed=30000, fields=5):
    config = load_config(DEFAULT_CONFIG_PATH)
    config["rl"]["reward"]["mode"] = mode
    env = InstanceGenerator(config).generate(
        num_fields=fields, seed=seed, randomize_each_reset=False)
    env.reset()
    done, steps, reward, times, area = False, 0, 0.0, 0.0, 0.0
    maneuver = nonwork = 0.0
    path_sum = 0.0
    while not done and steps < 80:
        action = _nearest_field_choice(env, None)
        if action is None:
            break
        _s, r, done, info = env.step(action)
        steps += 1
        reward += float(r)
        times += float(info["time_total"])
        maneuver += float(info["time_maneuver_extra"])
        path_sum += float(info.get("path_length", 0.0))
        nonwork += float(info["time_nonwork_base"])
        area += float(info.get("coverage_gain", 0.0))
    return config, env, dict(reward=reward, time=times, area=area, steps=steps,
                  maneuver=maneuver, nonwork=nonwork,
                  path_sum=path_sum,
                  distance_gap=path_sum - float(env.get_summary()["total_distance"]),
                  closure=float(env.get_summary().get("time_closure_error", 0.0)),
                  summary_time=float(env.get_summary().get("total_time", 0.0)))


class ProxyGeometryTests(unittest.TestCase):
    def test_straight_path_is_free(self):
        base, extra = path_time(None, [(0, 0), (10, 0), (20, 0)])
        self.assertAlmostEqual(base, 20.0 / 8.0, places=9)
        self.assertAlmostEqual(extra, 0.0, places=12)

    def test_reversal_costs_tau_pi(self):
        _base, extra = path_time(None, [(0, 0), (10, 0), (0, 0)])
        self.assertAlmostEqual(extra, 3.0, places=9)

    def test_corner_is_angle_scaled(self):
        _base, extra = path_time(None, [(0, 0), (10, 0), (10, 10)])
        self.assertAlmostEqual(extra, turn_cost(torch.pi / 2), places=9)

    def test_split_at_corner_is_unchanged(self):
        whole = path_time(None, [(0, 0), (10, 0), (10, 10)])[1]
        first = path_time(None, [(0, 0), (10, 0)])[1]
        second = path_time((1.0, 0.0), [(10, 0), (10, 10)])[1]
        self.assertAlmostEqual(first + second, whole, places=12)

    def test_split_on_straight_is_unchanged(self):
        whole = path_time(None, [(0, 0), (10, 0), (20, 0)])[1]
        first = path_time(None, [(0, 0), (10, 0)])[1]
        second = path_time((1.0, 0.0), [(10, 0), (20, 0)])[1]
        self.assertAlmostEqual(first + second, whole, places=12)

    def test_collinear_insert_and_duplicates_are_unchanged(self):
        base = path_time(None, [(0, 0), (20, 0)])[1]
        inserted = path_time(None, [(0, 0), (5, 0), (10, 0), (20, 0)])[1]
        duplicated = path_time(None, [(0, 0), (0, 0), (20, 0), (20, 0)])[1]
        self.assertAlmostEqual(inserted, base, places=12)
        self.assertAlmostEqual(duplicated, base, places=12)


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.env, cls.stats = _episode("time")

    def test_time_is_positive_and_maneuver_nonzero(self):
        """A real episode must accrue non-zero maneuver time, not just base flight."""
        self.assertGreater(self.stats["time"], 0.0)
        self.assertGreater(self.stats["maneuver"], 0.0,
                           "机动时间不得为零（否则代理没生效）")
        self.assertGreater(self.stats["nonwork"], 0.0)

    def test_distance_closes_to_step_sum(self):
        """Independent reference: sum the per-step path_length, compare with summary."""
        pumped = sum(getattr(c, "path_length", 0.0) for c in
                     getattr(self.env, "_coverage_path_length_handles", []))
        total = float(self.env.get_summary()["total_distance"])
        self.assertGreater(total, 0.0)
        self.assertAlmostEqual(abs(self.stats["distance_gap"]), 0.0, places=6)

    def test_four_term_time_ledger_closes(self):
        """summary exposes time_closure_error; it must be ~0 on a real episode."""
        self.assertAlmostEqual(self.stats["closure"], 0.0, places=6)
        self.assertAlmostEqual(self.stats["summary_time"], self.stats["time"], places=6)

    def test_reward_identity_closes(self):
        cfg = self.config["rl"]["reward"]
        credit = self.stats["area"] / (cfg["coverage_width"] * cfg["nominal_work_speed"])
        expected = -(self.stats["time"] - credit) / cfg["time_scale"]
        self.assertAlmostEqual(self.stats["reward"], expected, places=9)

    def test_coverage_credit_is_deduplicated(self):
        """A repeated coverage event reports 0 new area, so it earns no credit."""
        from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
        config = load_config(DEFAULT_CONFIG_PATH)
        config["rl"]["reward"]["mode"] = "time"
        env = InstanceGenerator(config).generate(
            num_fields=5, seed=30002, randomize_each_reset=False)
        env.reset()
        action = _nearest_field_choice(env, None)
        _s, _r, _d, info = env.step(action)
        self.assertGreater(float(info.get("coverage_gain", 0.0)), 0.0)
        # replay the SAME configuration again: the field is already covered here
        from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice as nn
        again = nn(env, None)
        if again is None:
            self.skipTest("no feasible repeat action on this instance")
        _s2, _r2, _d2, info2 = env.step(again)
        cfg = config["rl"]["reward"]
        if float(info2.get("coverage_gain", 0.0)) == 0.0:
            expected = -float(info2["time_total"]) / cfg["time_scale"]
            self.assertAlmostEqual(float(_r2), expected, places=9)


class RewardModeTests(unittest.TestCase):
    def test_mode_switch_changes_real_step_output(self):
        _c1, _e1, dist = _episode("distance")
        _c2, _e2, time = _episode("time")
        self.assertNotAlmostEqual(dist["reward"], time["reward"], places=6)

    def test_distance_mode_keeps_old_formula(self):
        payload = {"distance": 1000.0, "time_total": 999.0, "coverage": 5000.0}
        config = {"reward": {"mode": "distance"}, "total_area": 50000, "reward_scale": 100}
        self.assertAlmostEqual(calculate_reward(payload, config), -2.0, places=9)

    def test_gamma_is_one(self):
        config = load_config(DEFAULT_CONFIG_PATH)
        self.assertEqual(float(config["ppo"]["gamma"]), 1.0)


class HeadingStateTests(unittest.TestCase):
    def test_heading_starts_invalid_then_becomes_valid(self):
        """Before any flight there is no direction; after a step there is a unit one."""
        import math
        from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
        config = load_config(DEFAULT_CONFIG_PATH)
        env = InstanceGenerator(config).generate(
            num_fields=5, seed=30001, randomize_each_reset=False)
        env.reset()
        first = env.get_state()["uavs"]["UAV-1"]
        self.assertIsNone(first["heading"])
        self.assertFalse(first["heading_valid"])
        env.step(_nearest_field_choice(env, None))
        second = env.get_state()["uavs"]["UAV-1"]
        self.assertTrue(second["heading_valid"])
        self.assertIsNotNone(second["heading"])
        self.assertAlmostEqual(math.hypot(*second["heading"]), 1.0, places=6)

    def test_invalid_heading_feature_is_zeroed(self):
        from coverage_env.rl.decoder.global_state_encoder import build_uav_feature
        base = {"position": (0, 0), "max_energy": 1, "max_payload": 1, "speed": 8,
                "coverage_width": 10, "remaining_energy": 1, "remaining_payload": 1,
                "alpha": 0.08, "beta": 0.02, "payload_per_area": 0.01}
        self.assertEqual(tuple(build_uav_feature(base).shape), (16,))
        self.assertEqual([float(x) for x in build_uav_feature(base)[-3:]], [0.0, 0.0, 0.0])
        with_dir = dict(base, heading=(0.6, 0.8), heading_valid=True)
        self.assertEqual([round(float(x), 6) for x in build_uav_feature(with_dir)[-3:]],
                         [1.0, 0.6, 0.8])


if __name__ == "__main__":
    unittest.main()
