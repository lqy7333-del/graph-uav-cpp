"""Unit tests for simulator-backed action cost-to-go labels."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from coverage_env.rl.evaluation.cost_to_go_oracle import (
    evaluate_candidate_action_costs,
)


class _Action:
    def __init__(self, distance, *, terminal=False, invalid=False):
        self.distance = distance
        self.terminal = terminal
        self.invalid = invalid


class _Environment:
    def __init__(self, total_distance=10.0):
        self.total_distance = total_distance

    def get_summary(self):
        return {"total_distance": self.total_distance}

    def step(self, action):
        if action.invalid:
            raise ValueError("invalid transition")
        self.total_distance += action.distance
        done = action.terminal
        reason = "mission_complete" if done else None
        return {}, 0.0, done, {"success": True, "termination_reason": reason}


class CostToGoOracleTests(unittest.TestCase):
    def test_search_starts_at_successor_without_reset_and_uses_distance_delta(self):
        environment = _Environment(total_distance=10.0)
        action = _Action(2.0)

        def fake_beam_search(trial, **kwargs):
            self.assertFalse(kwargs["reset"])
            # The suffix adds 8 m after the 2 m immediate action.
            trial.total_distance += 8.0
            return ((-1.0, [], trial), 1)

        with patch(
            "coverage_env.rl.evaluation.cost_to_go_oracle.beam_search",
            side_effect=fake_beam_search,
        ):
            result = evaluate_candidate_action_costs(
                environment, [action], budget_seconds=1.0
            )[0]

        self.assertEqual(result.status, "completed")
        self.assertAlmostEqual(result.distance, 10.0)
        self.assertEqual(result.search_steps, 2)
        self.assertEqual(environment.total_distance, 10.0)

    def test_terminal_action_has_exact_cost_without_suffix_search(self):
        environment = _Environment(total_distance=25.0)
        result = evaluate_candidate_action_costs(
            environment, [_Action(7.5, terminal=True)], budget_seconds=1.0
        )[0]
        self.assertTrue(result.completed)
        self.assertAlmostEqual(result.distance, 7.5)
        self.assertEqual(result.search_steps, 1)

    def test_invalid_and_unresolved_are_not_given_fake_costs(self):
        environment = _Environment(total_distance=4.0)

        def no_completion(_trial, **_kwargs):
            return None, 3

        with patch(
            "coverage_env.rl.evaluation.cost_to_go_oracle.beam_search",
            side_effect=no_completion,
        ):
            invalid, unresolved = evaluate_candidate_action_costs(
                environment,
                [_Action(1.0, invalid=True), _Action(2.0)],
                budget_seconds=1.0,
            )
        self.assertTrue(invalid.status.startswith("invalid_action:"))
        self.assertIsNone(invalid.distance)
        self.assertEqual(unresolved.status, "no_complete_suffix_found")
        self.assertIsNone(unresolved.distance)
        self.assertEqual(environment.total_distance, 4.0)


if __name__ == "__main__":
    unittest.main()
