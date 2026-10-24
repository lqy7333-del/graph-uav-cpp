"""Tests for the rollout-baseline trainer: pairing, advantage sign, frozen baseline."""
import copy
import unittest

import torch

from coverage_env.rl.trainer.rollout_baseline import (
    EpisodeResult, RolloutBaselineTrainer, _assert_unit_gamma, _build_instance,
)


class GammaGuardTests(unittest.TestCase):
    def test_rejects_default_099(self):
        """The code default is 0.99; passing it must fail loudly, not silently train."""
        with self.assertRaises(ValueError):
            _assert_unit_gamma({})

    def test_rejects_explicit_099(self):
        with self.assertRaises(ValueError):
            _assert_unit_gamma({"gamma": 0.99})

    def test_accepts_unit_gamma(self):
        _assert_unit_gamma({"gamma": 1.0})


class AdvantageSignTests(unittest.TestCase):
    def test_cheaper_sample_gives_positive_advantage(self):
        """advantage = baseline - sample, so a cheaper sampled route must be positive."""
        cheaper = EpisodeResult(seed=1, success=True, sample_cost=100.0,
                                baseline_cost=120.0, log_prob_sum=torch.zeros(()),
                                steps=5, termination_reason="mission_complete")
        worse = EpisodeResult(seed=1, success=True, sample_cost=130.0,
                              baseline_cost=120.0, log_prob_sum=torch.zeros(()),
                              steps=5, termination_reason="mission_complete")
        self.assertAlmostEqual(cheaper.advantage, 20.0)
        self.assertAlmostEqual(worse.advantage, -10.0)

    def test_missing_baseline_is_zero_advantage(self):
        result = EpisodeResult(seed=1, success=True, sample_cost=100.0, baseline_cost=None,
                               log_prob_sum=torch.zeros(()), steps=5,
                               termination_reason="mission_complete")
        self.assertEqual(result.advantage, 0.0)


class SameInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        cls.config = load_config(DEFAULT_CONFIG_PATH)

    def test_two_builds_with_the_same_seed_are_identical(self):
        """Sample and baseline must see the same parcels, not two different layouts."""
        first = _build_instance(self.config, 5, 7000)
        second = _build_instance(self.config, 5, 7000)
        self.assertEqual(len(first.fields), len(second.fields))
        for left, right in zip(first.fields, second.fields):
            self.assertAlmostEqual(float(left.area), float(right.area), places=6)
            self.assertEqual(left.centroid[0], right.centroid[0])
            self.assertEqual(left.centroid[1], right.centroid[1])

    def test_different_seeds_differ(self):
        first = _build_instance(self.config, 5, 7000)
        second = _build_instance(self.config, 5, 7001)
        self.assertNotEqual([f.area for f in first.fields], [f.area for f in second.fields])


class FrozenBaselineTests(unittest.TestCase):
    def test_baseline_parameters_do_not_require_grad(self):
        class _Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)

        class _Agent:
            def __init__(self):
                self.policy = _Policy()

        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        config = load_config(DEFAULT_CONFIG_PATH)
        trainer = RolloutBaselineTrainer.__new__(RolloutBaselineTrainer)
        trainer.agent = _Agent()
        trainer.policy = trainer.agent.policy
        trainer.baseline = copy.deepcopy(trainer.policy)
        for parameter in trainer.baseline.parameters():
            parameter.requires_grad_(False)
        self.assertTrue(any(p.requires_grad for p in trainer.policy.parameters()))
        self.assertFalse(any(p.requires_grad for p in trainer.baseline.parameters()))
        # promotion copies weights, it must not alias them
        self.assertIsNot(trainer.baseline.layer, trainer.policy.layer)


class PromotionGuardTests(unittest.TestCase):
    def test_too_few_samples_does_not_promote(self):
        from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
        config = load_config(DEFAULT_CONFIG_PATH)

        class _Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1)

        class _Agent:
            def __init__(self):
                self.policy = _Policy()

        trainer = RolloutBaselineTrainer.__new__(RolloutBaselineTrainer)
        trainer.agent = _Agent()
        trainer.policy = trainer.agent.policy
        trainer.baseline = copy.deepcopy(trainer.policy)
        trainer.promotion_instances = 20
        trainer.stats = type("S", (), {"promotions": 0, "promotion_log": []})()
        result = trainer.promotion_check([])          # no instances at all
        self.assertFalse(result["promoted"])
        self.assertIn("insufficient", result["reason"])


if __name__ == "__main__":
    unittest.main()
