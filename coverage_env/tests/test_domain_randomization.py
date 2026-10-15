"""Cross-episode generalization and reward-normalization tests."""

from __future__ import annotations

import unittest

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.action_mask import get_action_mask


def _instance_signature(environment):
    """Return a hashable description of the current fields and UAV parameters."""
    fields = tuple(
        (field.field_id, round(field.area, 6), len(field.boundaries))
        for field in environment.fields
    )
    uavs = tuple(
        (
            round(uav.max_energy, 6), round(uav.max_payload, 6),
            round(uav.speed, 6), round(uav.coverage_width, 6),
            round(uav.alpha, 8), round(uav.beta, 8),
        )
        for uav in environment.uavs
    )
    return fields, uavs


class DomainRandomizationTests(unittest.TestCase):
    """Verify per-reset resampling, reproducibility, and reward scale."""

    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG_PATH)

    def test_reset_resamples_field_count_geometry_and_uavs(self):
        environment = InstanceGenerator(self.config).generate(seed=1234)
        limits = self.config["environment"]["num_fields"]
        signatures, areas, counts = [], [], []
        for _ in range(8):
            environment.reset()
            counts.append(len(environment.fields))
            areas.append(environment.total_area)
            signatures.append(_instance_signature(environment))
            self.assertGreaterEqual(counts[-1], int(limits["min"]))
            self.assertLessEqual(counts[-1], int(limits["max"]))
            expected_area = sum(
                point.point_area
                for field in environment.fields
                for point in field.grid_points
            )
            self.assertAlmostEqual(environment.total_area, expected_area)

        # The sampled instance really changes from episode to episode.
        self.assertGreater(len(set(signatures)), 1)
        self.assertGreater(len(set(areas)), 1)
        uav_parameters = {
            params
            for _fields, uavs in signatures
            for params in uavs
        }
        self.assertGreater(len(uav_parameters), 1)

    def test_same_seed_reproduces_the_episode_sequence(self):
        first = InstanceGenerator(self.config).generate(seed=99)
        second = InstanceGenerator(self.config).generate(seed=99)
        for _ in range(5):
            first.reset()
            second.reset()
            self.assertEqual(
                _instance_signature(first), _instance_signature(second)
            )
            self.assertAlmostEqual(first.total_area, second.total_area)

    def test_fixed_count_instance_is_not_resampled(self):
        environment = InstanceGenerator(self.config).generate(
            num_fields=5, seed=7
        )
        self.assertEqual(len(environment.fields), 5)
        first = _instance_signature(environment)
        environment.reset()
        self.assertEqual(len(environment.fields), 5)
        self.assertEqual(_instance_signature(environment), first)

    def test_step_reward_uses_the_current_episode_total_area(self):
        environment = InstanceGenerator(self.config).generate(
            num_fields=5, seed=7
        )
        environment.reset()
        actions = environment.get_action_candidates()
        mask = get_action_mask(actions, environment)
        action = next(action for action, ok in zip(actions, mask) if ok)
        _state, reward, done, info = environment.step(action)
        scale = float(self.config["rl"]["reward_scale"])
        expected = -info["path_length"] / environment.total_area * scale
        if not done:
            self.assertAlmostEqual(reward, expected, places=6)


if __name__ == "__main__":
    unittest.main()
