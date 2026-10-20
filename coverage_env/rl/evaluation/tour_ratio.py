"""Cheap, high-sample ordering metric: ``tour_ratio``.

Why this exists
---------------
The exhaustive reference (:mod:`coverage_env.rl.evaluation.order_reference`) costs
120 full environment episodes per instance, so it can only afford ~10 instances.
Measured on those 10, the Kendall tau standard error is **0.18** -- larger than
every effect the structural experiments were chasing (0.04 ... 0.24). Every
ordering conclusion drawn from it was therefore statistically empty.

``tour_ratio`` measures the same thing at a tiny fraction of the cost:

    tour_ratio = centroid_tour_length(policy visit order)
                 / centroid_tour_length(best tour over the same parcels)

Only *one* policy episode per instance is needed (no enumeration), so 100+
instances take minutes and the standard error drops to ~0.05. It deliberately
ignores entry/exit choice and resupply timing: this isolates **ordering**, which is
what the structural work is trying to improve. The centroid tour is a sound
reference because it scores within 0.49 of the exhaustive optimum on this task.

A ratio of 1.0 means the policy's order is as short as the best round trip;
1.1 means it flies 10% further than necessary.
"""

from __future__ import annotations

import argparse
import statistics
from itertools import permutations
from math import hypot

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.decoder.hierarchical_decoder import checkpoint_decoder_flags
from coverage_env.rl.ppo.ppo import PPOAgent

EXACT_LIMIT = 7


def _distance(first, second) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


def tour_length(order, centroids, depot) -> float:
    """Round-trip length depot -> order[0] -> ... -> order[-1] -> depot."""
    if not order:
        return 0.0
    total = _distance(depot, centroids[order[0]])
    total += sum(
        _distance(centroids[a], centroids[b]) for a, b in zip(order, order[1:])
    )
    return total + _distance(centroids[order[-1]], depot)


def _nearest_neighbour(field_ids, centroids, depot):
    remaining = list(field_ids)
    order, current = [], depot
    while remaining:
        closest = min(remaining, key=lambda name: _distance(current, centroids[name]))
        remaining.remove(closest)
        order.append(closest)
        current = centroids[closest]
    return order


def _local_search(order, centroids, depot, rounds: int = 12):
    """2-opt reversal + or-opt move on centroid distances (no environment)."""
    best = list(order)
    best_length = tour_length(best, centroids, depot)
    for _ in range(rounds):
        improved = False
        size = len(best)
        for i in range(size - 1):
            for j in range(i + 1, size):
                candidate = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                length = tour_length(candidate, centroids, depot)
                if length < best_length - 1e-9:
                    best, best_length, improved = candidate, length, True
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                candidate = best[:i] + best[i + 1:]
                candidate.insert(j, best[i])
                length = tour_length(candidate, centroids, depot)
                if length < best_length - 1e-9:
                    best, best_length, improved = candidate, length, True
        if not improved:
            break
    return best, best_length


def optimal_tour(field_ids, centroids, depot):
    """Exact for small instances, multi-start local search beyond ``EXACT_LIMIT``."""
    if len(field_ids) <= EXACT_LIMIT:
        best_order, best_length = None, float("inf")
        for permutation in permutations(field_ids):
            length = tour_length(permutation, centroids, depot)
            if length < best_length:
                best_order, best_length = list(permutation), length
        return best_order, best_length
    best_order = None
    best_length = float("inf")
    starts = [_nearest_neighbour(field_ids, centroids, depot)]
    starts += [list(field_ids), list(reversed(field_ids))]
    for start in starts:
        order, length = _local_search(start, centroids, depot)
        if length < best_length:
            best_order, best_length = order, length
    return best_order, best_length


def policy_visit_order(actor, environment) -> list[str]:
    """Field order the greedy policy actually visits (first visits, then the rest)."""
    state = environment.reset()
    order: list[str] = []
    with torch.no_grad():
        for _ in range(200):
            output = actor(state, environment=environment)
            distribution = output["action_distribution"]
            if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0:
                break
            action = CoveragePolicy.greedy_action(output)
            if action.field_id not in order:
                order.append(action.field_id)
            state, _reward, done, _info = environment.step(action)
            if done:
                break
    for field in environment.fields:
        if field.field_id not in order:
            order.append(field.field_id)
    return order


def build_actor(config, environment, checkpoint):
    """Build the policy with the decoder switches the checkpoint was trained with."""
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location="cpu")
        config["rl"]["decoder"].update(checkpoint_decoder_flags(saved["actor"]))
    torch.manual_seed(0)
    actor = CoveragePolicy(environment, config=config)
    if checkpoint is not None:
        PPOAgent(actor, config).load_checkpoint(checkpoint, map_location="cpu")
    actor.eval()
    return actor


def evaluate(config, checkpoint, instances: int, seed: int) -> dict:
    """Return mean/SE of ``tour_ratio`` over ``instances`` fresh episodes."""
    torch.manual_seed(0)
    environment = InstanceGenerator(config).generate(num_fields=5, seed=seed)
    actor = build_actor(config, environment, checkpoint)
    ratios, policy_orders, greedy_ratios = [], [], []
    for index in range(instances):
        instance = InstanceGenerator(config).generate(seed=seed + index)
        actor.environment = instance
        order = policy_visit_order(actor, instance)
        centroids = {field.field_id: field.centroid for field in instance.fields}
        depot = instance.depot.get_position()
        field_ids = list(centroids)
        length = tour_length(order, centroids, depot)
        _best_order, best_length = optimal_tour(field_ids, centroids, depot)
        if best_length <= 0.0:
            continue
        ratios.append(length / best_length)
        policy_orders.append(order)
        greedy = _nearest_neighbour(field_ids, centroids, depot)
        greedy_ratios.append(tour_length(greedy, centroids, depot) / best_length)
    if not ratios:
        return {"instances": 0}
    standard_error = (
        statistics.stdev(ratios) / (len(ratios) ** 0.5) if len(ratios) > 1 else 0.0
    )
    return {
        "instances": len(ratios),
        "ratio": statistics.mean(ratios),
        "stderr": standard_error,
        "exact": sum(1 for value in ratios if value <= 1.0001),
        "greedy_ratio": statistics.mean(greedy_ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, nargs="+", default=[None])
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9000)
    args = parser.parse_args()

    print(f"{'checkpoint':<44}{'n':>5}{'tour_ratio':>12}{'SE':>8}"
          f"{'optimal':>9}{'NN ratio':>10}")
    for checkpoint in args.checkpoint:
        config = load_config(DEFAULT_CONFIG_PATH)
        stats = evaluate(config, checkpoint, args.instances, args.seed)
        name = (checkpoint or "untrained").split("/")[-1]
        if checkpoint:
            name = f"{checkpoint.split('/')[-2]}/{name}"
        print(f"{name:<44}{stats['instances']:>5}{stats['ratio']:>12.4f}"
              f"{stats['stderr']:>8.4f}"
              f"{100.0 * stats['exact'] / max(1, stats['instances']):>8.0f}%"
              f"{stats['greedy_ratio']:>10.4f}")


if __name__ == "__main__":
    main()
