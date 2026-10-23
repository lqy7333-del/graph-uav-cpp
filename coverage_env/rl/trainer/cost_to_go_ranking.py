"""Collect simulator-search labels and train a pairwise action ranker.

The oracle evaluates complete continuations with the real environment. This
module only provides data collection and a supervised update interface; it does
not start training or alter the existing PPO loop.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from coverage_env.rl.evaluation.cost_to_go_oracle import (
    ActionCostEstimate,
    evaluate_candidate_action_costs,
)


@dataclass
class CostToGoRankingExample:
    """One state, its complete candidate set, and optional oracle labels.

    ``costs`` is aligned with ``actions`` and stores additional mission
    distance in meters. Unfeasible, unevaluated, and unresolved actions have a
    ``None`` cost and are excluded from pairwise comparisons.
    """

    observation: dict
    graph: Any
    actions: list
    feasible_mask: list[bool]
    costs: list[float | None]
    statuses: list[str]
    search_steps: list[int]
    total_area: float

    @property
    def labeled_action_count(self) -> int:
        """Number of feasible actions with a completed-search distance label."""
        return sum(
            feasible and cost is not None
            for feasible, cost in zip(self.feasible_mask, self.costs)
        )


def collect_cost_to_go_example(
    environment,
    policy,
    *,
    max_actions: int | None = 16,
    seed: int = 0,
    beam: int = 4,
    branches: int = 8,
    max_depth: int = 32,
    budget_seconds: float = 30.0,
) -> CostToGoRankingExample:
    """Snapshot a state and label a subset of feasible actions by search.

    Candidate evaluation is expensive: each selected action receives an
    independent, bounded suffix search. ``max_actions`` therefore defaults to
    a small uniform sample. Search failures remain unresolved labels rather
    than being mislabeled as impossible or assigned an arbitrary penalty.
    """
    observation = deepcopy(environment.get_state())
    graph = _snapshot_graph(policy.graph_builder.build_graph(environment))
    actions = list(environment.get_action_candidates())
    feasible_mask = [bool(value) for value in environment.get_action_mask(actions)]
    if len(feasible_mask) != len(actions):
        raise ValueError("Environment action mask does not align with candidates.")

    feasible_indices = [i for i, feasible in enumerate(feasible_mask) if feasible]
    if max_actions is not None:
        if max_actions <= 0:
            raise ValueError("max_actions must be positive or None.")
        if len(feasible_indices) > max_actions:
            # Sampling without replacement avoids the nearest-first bias of the
            # beam search's expansion heuristic when choosing teacher actions.
            feasible_indices = sorted(
                random.Random(seed).sample(feasible_indices, max_actions)
            )

    estimates: list[ActionCostEstimate] = evaluate_candidate_action_costs(
        environment,
        [actions[index] for index in feasible_indices],
        beam=beam,
        branches=branches,
        max_depth=max_depth,
        budget_seconds=budget_seconds,
    )
    if len(estimates) != len(feasible_indices):
        raise RuntimeError("Cost oracle returned a different number of labels than actions.")
    costs: list[float | None] = [None] * len(actions)
    statuses = ["infeasible" if not feasible else "not_evaluated"
                for feasible in feasible_mask]
    search_steps = [0] * len(actions)
    for index, estimate in zip(feasible_indices, estimates):
        if estimate.completed and estimate.distance is not None:
            costs[index] = float(estimate.distance)
        statuses[index] = estimate.status
        search_steps[index] = int(estimate.search_steps)

    total_area = float(getattr(environment, "total_area", 0.0))
    if total_area <= 0.0:
        total_area = sum(float(field.area) for field in environment.fields)

    return CostToGoRankingExample(
        observation=observation,
        graph=graph,
        actions=actions,
        feasible_mask=feasible_mask,
        costs=costs,
        statuses=statuses,
        search_steps=search_steps,
        total_area=max(total_area, 1e-12),
    )


def pairwise_cost_ranking_loss(
    scores: torch.Tensor,
    costs: Iterable[float | None],
    feasible_mask: Iterable[bool],
    *,
    total_area: float = 1.0,
    min_gap_per_area: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Make lower-cost actions score higher using a pairwise logistic loss.

    Costs are normalized by the same state's total coverable area. Pairs with
    unresolved/non-finite labels, infeasible actions, or a normalized cost gap
    no larger than ``min_gap_per_area`` are ignored. The returned zero-pair
    loss remains connected to ``scores`` so callers can safely inspect it.
    """
    flat_scores = scores.reshape(-1)
    cost_list = list(costs)
    mask_list = [bool(value) for value in feasible_mask]
    if flat_scores.numel() != len(cost_list) or len(mask_list) != len(cost_list):
        raise ValueError("scores, costs, and feasible_mask must have equal length.")
    if total_area <= 0.0:
        raise ValueError("total_area must be positive.")
    if min_gap_per_area < 0.0:
        raise ValueError("min_gap_per_area must be non-negative.")

    finite = torch.isfinite(flat_scores)
    eligible = []
    for index, (cost, feasible) in enumerate(zip(cost_list, mask_list)):
        if not feasible or cost is None or not finite[index]:
            continue
        value = float(cost)
        if value >= 0.0 and value < float("inf"):
            eligible.append((index, value / float(total_area)))

    pair_losses = []
    for left_pos in range(len(eligible)):
        left_index, left_cost = eligible[left_pos]
        for right_pos in range(left_pos + 1, len(eligible)):
            right_index, right_cost = eligible[right_pos]
            gap = abs(left_cost - right_cost)
            if gap <= min_gap_per_area:
                continue
            better, worse = (
                (left_index, right_index)
                if left_cost < right_cost
                else (right_index, left_index)
            )
            pair_losses.append(F.softplus(-(flat_scores[better] - flat_scores[worse])))

    if not pair_losses:
        return flat_scores.sum() * 0.0, 0
    return torch.stack(pair_losses).mean(), len(pair_losses)


class CostToGoRankingTrainer:
    """Apply explicit pairwise ranking updates to an existing CoveragePolicy."""

    def __init__(
        self,
        policy,
        *,
        learning_rate: float = 1e-4,
        min_gap_per_area: float = 0.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        if learning_rate <= 0.0 or max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive.")
        self.policy = policy
        self.min_gap_per_area = float(min_gap_per_area)
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    def loss_for_example(self, example: CostToGoRankingExample):
        """Score the saved action set and return ``(loss, pair_count, output)``."""
        output = self.policy(
            example.observation,
            graph=example.graph,
            candidate_actions=example.actions,
            feasible_mask=example.feasible_mask,
        )
        loss, pair_count = pairwise_cost_ranking_loss(
            output["scores"],
            example.costs,
            example.feasible_mask,
            total_area=example.total_area,
            min_gap_per_area=self.min_gap_per_area,
        )
        return loss, pair_count, output

    def update(self, example: CostToGoRankingExample) -> dict[str, float | int]:
        """Perform one supervised ranking update on a labeled state example."""
        loss, pair_count, _output = self.loss_for_example(example)
        if pair_count == 0:
            return {"ranking_loss": 0.0, "ranking_pairs": 0}
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.max_grad_norm
        )
        self.optimizer.step()
        return {
            "ranking_loss": float(loss.detach()),
            "ranking_pairs": pair_count,
        }


def _snapshot_graph(graph):
    """Clone graph tensors and move the stored search example to CPU."""
    if hasattr(graph, "clone"):
        snapshot = graph.clone()
        if hasattr(snapshot, "cpu"):
            snapshot = snapshot.cpu()
        return snapshot
    return deepcopy(graph)
