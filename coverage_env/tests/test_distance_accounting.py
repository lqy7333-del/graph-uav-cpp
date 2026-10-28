"""The per-step metadata must close to summary["total_distance"].

Historically ``info["path_length"]`` covered only the executed coverage segment, so the
resupply and final-return legs were invisible to any analysis that summed the per-step
values (the cost breakdown never closed, and gate-selection audits were biased). The
reward and total_distance were always correct; only the diagnostic metadata was short.
"""
import unittest

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice


class DistanceAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(DEFAULT_CONFIG_PATH)

    def _episode(self, seed, num_fields=5):
        env = InstanceGenerator(self.config).generate(
            num_fields=num_fields, seed=seed, randomize_each_reset=False
        )
        env.reset()
        steps, reward, done = [], 0.0, False
        while not done and len(steps) < 400:
            action = _nearest_field_choice(env, None)
            if action is None:
                break
            _s, r, done, info = env.step(action)
            reward += float(r)
            steps.append(info)
        return env, steps, reward

    def test_step_path_lengths_sum_to_total_distance(self):
        env, steps, _reward = self._episode(21013)
        total = float(env.get_summary()["total_distance"])
        summed = sum(float(s.get("path_length", 0.0)) for s in steps)
        self.assertAlmostEqual(summed, total, places=3,
                               msg="Σ info['path_length'] 未闭合到 total_distance")

    def test_components_sum_to_total_distance(self):
        env, steps, _reward = self._episode(21013)
        total = float(env.get_summary()["total_distance"])
        coverage = sum(float(s.get("coverage_only_distance", 0.0)) for s in steps)
        transit = sum(float(s.get("transit_distance", 0.0)) for s in steps)
        legs = sum(float(s.get("resupply_or_return_distance", 0.0)) for s in steps)
        self.assertAlmostEqual(coverage + transit + legs, total, places=3,
                               msg="覆盖 + 转场 + 补给/返航 未闭合到 total_distance")

    def test_reward_still_matches_total_distance(self):
        """The fix must not change the reward: it already covered every leg."""
        env, _steps, reward = self._episode(21013)
        total_area = float(env.total_area)
        scale = float(self.config["rl"].get("reward_scale", 100.0))
        self.assertAlmostEqual(-reward * total_area / scale,
                               float(env.get_summary()["total_distance"]), places=2)


if __name__ == "__main__":
    unittest.main()
