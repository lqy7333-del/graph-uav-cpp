"""Unit tests for simulator-backed action cost-to-go labels."""

from __future__ import annotations

import unittest
from unittest.mock import patch
from dataclasses import dataclass
from types import SimpleNamespace

from coverage_env.rl.evaluation.cost_to_go_oracle import (
    evaluate_candidate_action_costs,
)
from coverage_env.rl.evaluation.beam_oracle import beam_search, branch_actions
from coverage_env.rl.trainer.cost_to_go_ranking import (
    collect_beam_cost_to_go_examples,
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


@dataclass(frozen=True)
class _SpatialAction:
    field_id: str
    entry_boundary_id: str
    exit_boundary_id: str
    distance: float


class _TreeEnvironment:
    """Two-action miniature tree with known complete suffix costs."""

    def __init__(self):
        self.total_area = 100.0
        self.fields = [
            SimpleNamespace(
                field_id=field_id,
                area=50.0,
                boundaries=[
                    SimpleNamespace(id="B0", center=(0.0, 0.0)),
                    SimpleNamespace(id="B1", center=(1.0, 0.0)),
                ],
            )
            for field_id in ("F0", "F1", "F2")
        ]
        self.depot = SimpleNamespace(get_position=lambda: (0.0, 0.0))
        self.uavs = [SimpleNamespace(current_position=(0.0, 0.0))]
        self.reset()

    def reset(self):
        self.depth = 0
        self.total_distance = 0.0
        self.chosen_field = None
        return self.get_state()

    def get_state(self):
        return {"depth": self.depth, "chosen_field": self.chosen_field}

    def get_summary(self):
        return {"total_distance": self.total_distance, "coverage_ratio": 0.5}

    def get_action_candidates(self):
        if self.depth == 0:
            return [
                _SpatialAction("F0", "B0", "B1", 1.0),
                _SpatialAction("F1", "B0", "B1", 5.0),
                _SpatialAction("F2", "B0", "B1", 7.0),
            ]
        if self.depth == 1:
            return [_SpatialAction(self.chosen_field, "B0", "B0", 2.0)]
        return []

    def get_action_mask(self, actions=None):
        return [True] * len(self.get_action_candidates() if actions is None else actions)

    def get_field(self, field_id):
        return next(field for field in self.fields if field.field_id == field_id)

    def step(self, action):
        self.total_distance += action.distance
        self.depth += 1
        self.chosen_field = action.field_id
        done = self.depth == 2
        return self.get_state(), -action.distance, done, {
            "success": True,
            "termination_reason": "mission_complete" if done else None,
        }


class CostToGoOracleTests(unittest.TestCase):
    def test_diverse_branch_selection_samples_across_field_distance_range(self):
        fields = []
        actions = []
        for index in range(5):
            field_id = f"F{index}"
            fields.append(SimpleNamespace(
                field_id=field_id,
                boundaries=[SimpleNamespace(id="B0", center=(float(index), 0.0))],
            ))
            actions.append(_SpatialAction(field_id, "B0", "B0", 1.0))

        class Environment:
            uavs = [SimpleNamespace(current_position=(0.0, 0.0))]
            depot = SimpleNamespace(get_position=lambda: (0.0, 0.0))

            def get_action_candidates(self):
                return actions

            def get_action_mask(self, _actions):
                return [True] * len(actions)

            def get_field(self, field_id):
                return next(field for field in fields if field.field_id == field_id)

        selected = branch_actions(Environment(), 3, diverse_fields=True)
        self.assertEqual([action.field_id for action in selected], ["F0", "F2", "F4"])

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

    def test_one_trace_search_recovers_pairwise_costs_at_the_same_state(self):
        environment = _TreeEnvironment()

        class GraphBuilder:
            def build_graph(self, state):
                return {"depth": state.depth, "chosen_field": state.chosen_field}

        policy = SimpleNamespace(graph_builder=GraphBuilder())
        best, steps, examples, diagnostics = collect_beam_cost_to_go_examples(
            environment,
            policy,
            beam=2,
            branches=2,
            max_depth=3,
            budget_seconds=2.0,
        )

        root_example = next(example for example in examples
                            if example.observation["depth"] == 0)
        self.assertEqual(root_example.costs, [3.0, None, 9.0])
        self.assertEqual(root_example.statuses[1], "branch_pruned")
        self.assertEqual(root_example.labeled_action_count, 2)
        self.assertEqual(diagnostics["completed_paths"], 2)
        self.assertEqual(diagnostics["rankable_contexts"], 1)
        self.assertEqual(diagnostics["pair_count"], 1)
        self.assertEqual(steps, 4)
        self.assertIsNotNone(best)
        self.assertEqual(environment.total_distance, 0.0)

    def test_beam_search_legacy_return_shape_is_unchanged(self):
        result = beam_search(
            _TreeEnvironment(),
            beam=2,
            branches=2,
            max_depth=3,
            budget_seconds=2.0,
            verbose=False,
        )
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])


if __name__ == "__main__":
    unittest.main()
