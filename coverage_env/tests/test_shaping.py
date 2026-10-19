"""Potential-based shaping: the discounted return must only shift by a constant.

That is the whole safety argument (Ng, Harada & Russell 1999): with
``Phi(terminal) = 0`` the shaped return of *any* trajectory equals the true return
plus ``-Phi(s_0)``, which does not depend on the policy, so the optimal policy is
unchanged. These tests verify the identity directly rather than trusting it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.shaping import (
    potential,
    remaining_tour_length,
    shaped_reward,
)


class _NoWork:
    """Minimal stand-in for a finished environment."""

    fields = [SimpleNamespace(get_uncovered_points=lambda: [])]
    uavs = []
    depot = SimpleNamespace(get_position=lambda: (0.0, 0.0))


class ShapingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = load_config(DEFAULT_CONFIG_PATH)
        self.environment = InstanceGenerator(self.config).generate(
            num_fields=5, seed=1005, randomize_each_reset=False
        )
        self.area = self.environment.total_area

    def test_potential_is_zero_when_no_work_is_left(self):
        """The terminal condition is what makes the telescoping exact."""
        self.assertEqual(remaining_tour_length(_NoWork()), 0.0)
        self.assertEqual(potential(_NoWork(), 1.0, self.area), 0.0)

    def test_potential_is_negative_and_shrinks_with_work(self):
        self.environment.reset()
        before = potential(self.environment, 1.0, self.area)
        self.assertLess(before, 0.0)
        self.assertGreater(remaining_tour_length(self.environment), 0.0)

    def test_shaped_return_differs_from_true_return_by_a_constant(self):
        """Run real episodes and check ``G' = G - Phi(s_0)`` in every one."""
        gamma = 0.99
        strength = 1.0
        policy = CoveragePolicy(self.environment, config=self.config)
        policy.eval()
        for trial in range(3):
            state = self.environment.reset()
            phi = potential(self.environment, strength, self.area)
            phi_start = phi
            true_return = 0.0
            shaped_return = 0.0
            discount = 1.0
            with torch.no_grad():
                for _ in range(40):
                    output = policy(state, environment=self.environment)
                    distribution = output["action_distribution"]
                    if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0:
                        break
                    index = int(torch.multinomial(distribution, 1).item())
                    action = output["actions"][index]
                    state, reward, done, _info = self.environment.step(action)
                    phi_next = potential(self.environment, strength, self.area)
                    true_return += discount * float(reward)
                    shaped_return += discount * shaped_reward(
                        reward, phi, phi_next, gamma
                    )
                    discount *= gamma
                    phi = phi_next
                    if done:
                        break
            # Phi is 0 at termination, so the only difference is -Phi(s_0).
            self.assertAlmostEqual(
                shaped_return - true_return, -phi_start, places=6,
                msg=f"trial {trial}: the shaping is not a constant offset",
            )


if __name__ == "__main__":
    unittest.main()
