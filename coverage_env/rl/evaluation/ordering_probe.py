"""Diagnose *why* the policy cannot order: representation, optimiser, or reward.

The question this answers
------------------------
``tour_ratio`` shows every trained policy orders worse than a nearest-neighbour
heuristic (1.32~1.40 against 1.15), while still beating that heuristic on total
reward. Is the policy unable to *represent* a good order, or unable to *learn* one?

Method
------
Collect (state -> candidate contexts) from rollouts of the trained policy, label
each step with the **centroid-optimal next parcel** (a tour computed by nearest
neighbour + 2-opt from the UAV's current position over the unfinished parcels),
and then fit a small supervised head on exactly the features the decoder sees.
This is a *diagnostic probe*, not a training method: it measures whether the
information is present in the representation.

Interpretation
--------------
* probe accuracy >> policy accuracy  -> the features are sufficient and the
  policy-gradient path failed to find it (optimisation / reward / credit).
* probe accuracy ~  policy accuracy  -> the representation does not contain the
  information (encoder / feature design).
* linear probe already high          -> the information is easily accessible;
  the failure is purely in how the head is trained.
* only the MLP probe high            -> the information is there but needs the
  non-linear head to extract, i.e. the policy's loss landscape is the obstacle.

Usage
-----
    PYTHONPATH=. python -m coverage_env.rl.evaluation.ordering_probe \
        --checkpoint coverage_env/results/run_hier_v5/ckpt_update_100.pt \
        --episodes 80
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn as nn

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.decoder.hierarchical_decoder import checkpoint_decoder_flags
from coverage_env.rl.ppo.ppo import PPOAgent

MAX_STEPS = 60


def _distance(first, second) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _tour_length(order, centroids, start) -> float:
    if not order:
        return 0.0
    total = _distance(start, centroids[order[0]])
    total += sum(
        _distance(centroids[a], centroids[b]) for a, b in zip(order, order[1:])
    )
    return total


def _tour_from(order, centroids, start):
    best, best_length = list(order), _tour_length(order, centroids, start)
    for _ in range(8):
        improved = False
        size = len(best)
        for i in range(size - 1):
            for j in range(i + 1, size):
                candidate = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                length = _tour_length(candidate, centroids, start)
                if length < best_length - 1e-9:
                    best, best_length, improved = candidate, length, True
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                candidate = best[:i] + best[i + 1:]
                candidate.insert(j, best[i])
                length = _tour_length(candidate, centroids, start)
                if length < best_length - 1e-9:
                    best, best_length, improved = candidate, length, True
        if not improved:
            break
    return best, best_length


def _nearest_neighbour_order(field_ids, centroids, start):
    remaining = list(field_ids)
    order, current = [], start
    while remaining:
        closest = min(remaining, key=lambda name: _distance(current, centroids[name]))
        remaining.remove(closest)
        order.append(closest)
        current = centroids[closest]
    return order


def best_next_field(environment) -> str | None:
    """Centroid-optimal next parcel from the UAV's current position."""
    remaining = [
        field.field_id for field in environment.fields
        if field.get_uncovered_points()
    ]
    if not remaining:
        return None
    if len(remaining) == 1:
        return remaining[0]
    centroids = {field.field_id: field.centroid for field in environment.fields}
    position = environment.uavs[0].current_position or environment.depot.get_position()
    start = _nearest_neighbour_order(remaining, centroids, position)
    order, _length = _tour_from(start, centroids, position)
    return order[0]


def collect(actor, config, episodes: int, seed: int) -> dict:
    """Roll out the policy and record candidate contexts with ordering labels."""
    torch.manual_seed(0)
    random.seed(0)
    contexts, groups, targets, policy_fields = [], [], [], []
    group = 0
    for index in range(episodes):
        environment = InstanceGenerator(config).generate(seed=seed + index)
        state = environment.reset()
        for _ in range(MAX_STEPS):
            output = actor(state, environment=environment)
            distribution = output["action_distribution"]
            mask = output["mask"]
            if distribution.numel() == 0 or not bool(mask.any()):
                break
            label = best_next_field(environment)
            if label is None:
                break
            actions = output["actions"]
            chosen = CoveragePolicy.greedy_action(output)
            centroids = {
                field.field_id: field.centroid for field in environment.fields
            }
            position = (
                environment.uavs[0].current_position
                or environment.depot.get_position()
            )
            remaining = [
                field.field_id for field in environment.fields
                if field.get_uncovered_points()
            ]
            nearest = min(
                remaining, key=lambda name: _distance(position, centroids[name])
            )
            contexts.append(output["action_input"].detach())
            groups.append(
                torch.full((len(actions),), group, dtype=torch.long)
            )
            targets.append(
                torch.tensor([action.field_id == label for action in actions])
            )
            policy_fields.append((chosen.field_id, label, nearest, len(remaining)))
            group += 1
            state, _reward, done, _info = environment.step(chosen)
            if done:
                break
    return {
        "context": torch.cat(contexts),
        "group": torch.cat(groups),
        "target": torch.cat(targets),
        "policy_field": [row[0] for row in policy_fields],
        "best_field": [row[1] for row in policy_fields],
        "nearest_field": [row[2] for row in policy_fields],
        "remaining": [row[3] for row in policy_fields],
        "groups": group,
    }


def _step_accuracy(scores, group, target, num_groups: int) -> float:
    """Fraction of steps whose highest-scoring candidate has the target field."""
    hits = 0
    for index in range(num_groups):
        rows = group == index
        if not bool(rows.any()):
            continue
        values = scores[rows]
        best = int(torch.argmax(values).item())
        hits += int(bool(target[rows][best]))
    return hits / max(1, num_groups)


class Head(nn.Module):
    """Per-candidate scorer, either linear or a small MLP."""

    def __init__(self, input_dim: int, kind: str, hidden: int = 256) -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(input_dim, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
            )

    def forward(self, features):
        return self.net(features).squeeze(-1)


def _segment_logsumexp(values, segments, num_segments: int):
    """log(sum(exp(values))) per segment, vectorised and numerically stable."""
    maxima = torch.full((num_segments,), float("-inf"))
    maxima = maxima.scatter_reduce(0, segments, values, reduce="amax")
    maxima = torch.where(torch.isinf(maxima), torch.zeros_like(maxima), maxima)
    shifted = torch.exp(values - maxima[segments])
    totals = torch.zeros(num_segments).index_add_(0, segments, shifted)
    return maxima + torch.log(totals.clamp_min(1e-12))


def fit_probe(data, kind: str, *, steps: int = 600, lr: float = 1e-2) -> float:
    """Train a supervised head on the recorded features; return step accuracy.

    Objective: per step, ``-log( sum of probability on the target field's
    candidates )``. Every step offers several candidates for the same parcel, so
    the target is a *set*, and matching any of them is a success.
    """
    head = Head(data["context"].shape[1], kind)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    context, group, target = data["context"], data["group"], data["target"]
    num_groups = data["groups"]
    positive_counts = torch.zeros(num_groups).index_add_(
        0, group, target.float()
    )
    supervised = positive_counts > 0
    for _step in range(steps):
        scores = head(context)
        all_lse = _segment_logsumexp(scores, group, num_groups)
        positive_scores = scores.masked_fill(~target, float("-inf"))
        positive_lse = _segment_logsumexp(positive_scores, group, num_groups)
        per_step = (all_lse - positive_lse).masked_fill(~supervised, 0.0)
        loss = per_step.sum() / supervised.sum().clamp_min(1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return _step_accuracy(head(context), group, target, num_groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--probe-steps", type=int, default=600)
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu")
        config["rl"]["decoder"].update(checkpoint_decoder_flags(saved["actor"]))
    torch.manual_seed(0)
    environment = InstanceGenerator(config).generate(num_fields=5, seed=args.seed)
    actor = CoveragePolicy(environment, config=config)
    if args.checkpoint:
        PPOAgent(actor, config).load_checkpoint(args.checkpoint, map_location="cpu")
    actor.eval()

    data = collect(actor, config, args.episodes, args.seed)
    steps = data["groups"]
    policy_hits = sum(
        1 for chosen, best in zip(data["policy_field"], data["best_field"])
        if chosen == best
    )
    nearest_hits = sum(
        1 for nearest, best in zip(data["nearest_field"], data["best_field"])
        if nearest == best
    )
    chance = sum(1.0 / max(1, n) for n in data["remaining"]) / max(1, steps)
    print(f"steps={steps}  candidates={data['context'].shape[0]}  "
          f"context_dim={data['context'].shape[1]}")
    print(f"  chance                     {100.0 * chance:6.1f}%")
    print(f"  nearest-field heuristic    {100.0 * nearest_hits / max(1, steps):6.1f}%")
    print(f"  trained policy             {100.0 * policy_hits / max(1, steps):6.1f}%")
    print(f"  supervised linear probe    "
          f"{100.0 * fit_probe(data, 'linear', steps=args.probe_steps):6.1f}%")
    print(f"  supervised MLP probe       "
          f"{100.0 * fit_probe(data, 'mlp', steps=args.probe_steps):6.1f}%")


if __name__ == "__main__":
    main()
