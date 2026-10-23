"""Unit tests for pairwise cost-to-go ranking and saved policy contexts."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from coverage_env.rl.trainer.cost_to_go_ranking import (
    CostToGoRankingExample,
    CostToGoRankingTrainer,
    collect_cost_to_go_example,
    pairwise_cost_ranking_loss,
)
from coverage_env.rl.evaluation.cost_to_go_oracle import ActionCostEstimate


class _ScorePolicy(torch.nn.Module):
    """Tiny differentiable test double; no optimizer update is run here."""

    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor([0.0, 0.0, 9.0]))

    def forward(self, observation, *, graph, candidate_actions, feasible_mask):
        self.assertions = (
            observation["state_id"], graph, candidate_actions, feasible_mask
        )
        return {"scores": self.values.reshape(-1, 1)}


class CostToGoRankingTests(unittest.TestCase):
    def test_lower_cost_should_receive_higher_score(self):
        good_order = torch.tensor([[1.0], [-1.0], [500.0]], requires_grad=True)
        bad_order = torch.tensor([[-1.0], [1.0], [500.0]], requires_grad=True)
        costs = [10.0, 20.0, None]
        mask = [True, True, False]
        good_loss, pairs = pairwise_cost_ranking_loss(
            good_order, costs, mask, total_area=100.0
        )
        bad_loss, bad_pairs = pairwise_cost_ranking_loss(
            bad_order, costs, mask, total_area=100.0
        )
        self.assertEqual(pairs, 1)
        self.assertEqual(bad_pairs, 1)
        self.assertLess(float(good_loss), float(bad_loss))
        good_loss.backward()
        self.assertLess(float(good_order.grad[0]), 0.0)
        self.assertGreater(float(good_order.grad[1]), 0.0)
        self.assertEqual(float(good_order.grad[2]), 0.0)

    def test_unresolved_infeasible_and_near_tie_labels_are_ignored(self):
        scores = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)
        loss, pairs = pairwise_cost_ranking_loss(
            scores,
            [5.0, None, 5.00001],
            [True, True, True],
            total_area=100.0,
            min_gap_per_area=1e-5,
        )
        self.assertEqual(pairs, 0)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))

    def test_saved_example_uses_existing_policy_forward_without_update(self):
        policy = _ScorePolicy()
        trainer = CostToGoRankingTrainer(policy, learning_rate=1e-4)
        example = CostToGoRankingExample(
            observation={"state_id": "s0"},
            graph=object(),
            actions=["a", "b", "c"],
            feasible_mask=[True, True, False],
            costs=[2.0, 4.0, None],
            statuses=["completed", "completed", "infeasible"],
            search_steps=[3, 4, 0],
            total_area=20.0,
        )
        loss, pairs, _ = trainer.loss_for_example(example)
        self.assertEqual(pairs, 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(policy.assertions[0], "s0")
        self.assertIs(policy.assertions[1], example.graph)
        self.assertEqual(policy.assertions[2], example.actions)
        self.assertEqual(policy.assertions[3], example.feasible_mask)
        # loss_for_example only constructs the objective. It does not run an
        # optimizer step or launch any training loop.
        self.assertEqual(policy.values.detach().tolist(), [0.0, 0.0, 9.0])

    def test_collection_keeps_labels_aligned_and_does_not_mutate_environment(self):
        actions = ["a", "b", "c"]

        class Environment:
            fields = [SimpleNamespace(area=120.0)]
            total_area = 100.0

            def get_state(self):
                return {"state_id": "root"}

            def get_action_candidates(self):
                return actions

            def get_action_mask(self, _actions):
                return [True, False, True]

        class GraphBuilder:
            def build_graph(self, _environment):
                return {"graph_state": "root"}

        class Policy:
            graph_builder = GraphBuilder()

        estimates = [
            ActionCostEstimate("a", 12.0, "completed", 4),
            ActionCostEstimate("c", None, "no_complete_suffix_found", 8),
        ]
        with patch(
            "coverage_env.rl.trainer.cost_to_go_ranking.evaluate_candidate_action_costs",
            return_value=estimates,
        ) as evaluate:
            example = collect_cost_to_go_example(
                Environment(), Policy(), max_actions=None
            )

        evaluate.assert_called_once()
        self.assertEqual(example.actions, actions)
        self.assertEqual(example.feasible_mask, [True, False, True])
        self.assertEqual(example.costs, [12.0, None, None])
        self.assertEqual(example.statuses,
                         ["completed", "infeasible", "no_complete_suffix_found"])
        self.assertEqual(example.search_steps, [4, 0, 8])
        self.assertEqual(example.total_area, 100.0)


if __name__ == "__main__":
    unittest.main()
