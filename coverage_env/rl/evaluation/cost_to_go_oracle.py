"""Estimate candidate-action cost-to-go with the real coverage environment.

Each candidate is applied to an isolated copy of the current environment. A
bounded beam search then completes the mission from that successor state. The
reported distance is the best *found feasible completion*, not a claim of a
globally optimal cost-to-go.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

from coverage_env.rl.evaluation.beam_oracle import beam_search


@dataclass(frozen=True)
class ActionCostEstimate:
    """Search result for one candidate action.

    ``distance`` is the additional mission distance from the queried state to
    mission completion. It is ``None`` when the action was invalid or bounded
    search did not find a complete continuation; those cases must not be
    treated as proof that the action is impossible.
    """

    action: Any
    distance: float | None
    status: str
    search_steps: int = 0

    @property
    def completed(self) -> bool:
        """Whether search found a complete, feasible continuation."""
        return self.status == "completed"


def evaluate_candidate_action_costs(
    environment,
    actions: Iterable[Any],
    *,
    beam: int = 4,
    branches: int = 8,
    max_depth: int = 32,
    budget_seconds: float = 30.0,
) -> list[ActionCostEstimate]:
    """Estimate long-term cost for each action from the exact current state.

    The same beam width, branch limit, depth, and time budget are used for each
    candidate. The source environment is never mutated. ``budget_seconds`` is a
    per-action suffix-search budget; callers should first profile a small set
    of states before constructing a large labeling dataset.
    """
    if beam <= 0 or branches <= 0 or max_depth <= 0:
        raise ValueError("beam, branches, and max_depth must be positive.")
    if budget_seconds <= 0:
        raise ValueError("budget_seconds must be positive.")

    action_list = list(actions)
    initial_distance = _total_distance(environment)
    estimates: list[ActionCostEstimate] = []

    for action in action_list:
        trial = deepcopy(environment)
        try:
            _next_state, _reward, done, info = trial.step(action)
        except (KeyError, ValueError, TypeError, RuntimeError, IndexError) as exc:
            estimates.append(ActionCostEstimate(
                action=action,
                distance=None,
                status=f"invalid_action:{type(exc).__name__}",
            ))
            continue

        if info.get("success") is False:
            estimates.append(ActionCostEstimate(
                action=action, distance=None, status="invalid_action"
            ))
            continue

        if done:
            if info.get("termination_reason") == "mission_complete":
                estimates.append(ActionCostEstimate(
                    action=action,
                    distance=_distance_delta(trial, initial_distance),
                    status="completed",
                    search_steps=1,
                ))
            else:
                estimates.append(ActionCostEstimate(
                    action=action,
                    distance=None,
                    status="terminated_without_completion",
                    search_steps=1,
                ))
            continue

        best, suffix_steps = beam_search(
            trial,
            beam=beam,
            branches=branches,
            max_depth=max_depth,
            budget_seconds=budget_seconds,
            verbose=False,
            reset=False,
        )
        if best is None:
            estimates.append(ActionCostEstimate(
                action=action,
                distance=None,
                status="no_complete_suffix_found",
                search_steps=1 + suffix_steps,
            ))
            continue

        final_environment = best[2]
        distance = _distance_delta(final_environment, initial_distance)
        estimates.append(ActionCostEstimate(
            action=action,
            distance=distance,
            status="completed" if isfinite(distance) else "invalid_cost",
            search_steps=1 + suffix_steps,
        ))

    return estimates


def _total_distance(environment) -> float:
    summary = environment.get_summary()
    return float(summary["total_distance"])


def _distance_delta(environment, initial_distance: float) -> float:
    delta = _total_distance(environment) - initial_distance
    # Floating-point accumulation can produce a tiny negative delta for a
    # no-op terminal transition; physical mission distance cannot be negative.
    return max(0.0, float(delta))
