"""Reference for the field-ordering problem: enumerate every field order.

The nearest-field heuristic has been overtaken by the trained policy, so it no
longer tells us how much room is left. This builds a stronger achievable
reference on small instances (5 parcels -> 5! = 120 orders): it replays the real
environment once per order, choosing entry/exit with a fixed rule.

What the number means
---------------------
It is an **achievable** score for this instance under a fixed entry/exit rule, so
it is a *lower* bound on the ordering headroom, never the true optimum: a
different entry/exit choice for the same order could score higher. If the trained
policy already matches it, ordering is not where the remaining cost is.

Usage
-----
    PYTHONPATH=. python -m coverage_env.rl.evaluation.order_reference \
        --checkpoint coverage_env/results/run_x/ckpt_update_400.pt \
        --seeds 1005 1006 1007
"""

from __future__ import annotations

import argparse
from itertools import permutations
from math import hypot

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent


def _distance(first, second) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])


MAX_STEPS = 80


def _choose_action(environment, field_id, next_target):
    """Pick the feasible (field, entry, exit) that best fits the fixed rule.

    Rule: cheapest ``distance(uav, entry) + distance(exit, next_target)``, which is
    the natural "enter from where I am, leave towards where I go next" choice.
    Applying the *same* rule to every order keeps the comparison about ordering.
    """
    actions = environment.get_action_candidates()
    mask = environment.get_action_mask(actions)
    position = environment.uavs[0].current_position
    field = environment.get_field(field_id)
    centres = {boundary.id: boundary.center for boundary in field.boundaries}
    best, best_cost = None, float("inf")
    for action, feasible in zip(actions, mask):
        if not feasible or action.field_id != field_id:
            continue
        entry = centres.get(action.entry_boundary_id)
        exit_point = centres.get(action.exit_boundary_id)
        if entry is None or exit_point is None:
            continue
        cost = _distance(position, entry) + _distance(exit_point, next_target)
        if cost < best_cost:
            best, best_cost = action, cost
    return best


def score_order(environment, order):
    """Run one full episode visiting the parcels in ``order``; return the reward.

    Returns ``None`` when the order cannot be executed (no feasible action).
    """
    environment.reset()
    total = 0.0
    done = False
    steps = 0
    for index, field_id in enumerate(order):
        next_field = order[index + 1] if index + 1 < len(order) else None
        next_target = (
            environment.get_field(next_field).centroid if next_field is not None
            else environment.depot.get_position()
        )
        action = _choose_action(environment, field_id, next_target)
        if action is None:
            return None
        _state, reward, done, info = environment.step(action)
        total += float(reward)
        steps += 1
        if done:
            break

    # A single visit does not necessarily finish a parcel (the environment picks
    # the workload), so finish whatever is left with the same greedy rule. Without
    # this the enumeration could "win" by leaving work undone: the reward is
    # distance per unit area, so an incomplete episode skips both the extra
    # coverage distance and the final return leg.
    while not done and steps < MAX_STEPS:
        target = _nearest_unfinished_field(environment)
        if target is None:
            return None
        action = _choose_action(environment, target, environment.depot.get_position())
        if action is None:
            return None
        _state, reward, done, info = environment.step(action)
        total += float(reward)
        steps += 1
    if not done or info.get("termination_reason") != "mission_complete":
        return None
    return total


def _nearest_unfinished_field(environment):
    """Return the id of the nearest parcel that still has uncovered points."""
    position = environment.uavs[0].current_position
    best, best_distance = None, float("inf")
    for field in environment.fields:
        if not field.get_uncovered_points():
            continue
        distance = _distance(position, field.centroid)
        if distance < best_distance:
            best, best_distance = field.field_id, distance
    return best


def exhaustive_reference(environment, field_ids):
    """Return the best/median/worst reward over every permutation of ``field_ids``."""
    scores = []
    for order in permutations(field_ids):
        value = score_order(environment, order)
        if value is not None:
            scores.append(value)
    scores.sort()
    return scores


def _policy_reward(actor, environment) -> float:
    """Greedy policy episode reward on this instance."""
    state = environment.reset()
    total = 0.0
    with torch.no_grad():
        for _ in range(200):
            output = actor(state, environment=environment)
            distribution = output["action_distribution"]
            if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0:
                break
            action = output["actions"][int(distribution.argmax().item())]
            state, reward, done, _info = environment.step(action)
            total += float(reward)
            if done:
                break
    return total


def _nearest_field_reward(environment) -> float:
    """Nearest-unfinished-field heuristic episode reward on this instance."""
    state = environment.reset()
    total = 0.0
    for _ in range(200):
        actions = environment.get_action_candidates()
        if not actions:
            break
        mask = environment.get_action_mask(actions)
        position = environment.uavs[0].current_position
        best, best_distance = None, float("inf")
        for action, feasible in zip(actions, mask):
            if not feasible:
                continue
            field = environment.get_field(action.field_id)
            distance = _distance(position, field.centroid)
            if distance < best_distance:
                best, best_distance = action, distance
        if best is None:
            break
        state, reward, done, _info = environment.step(best)
        total += float(reward)
        if done:
            break
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-fields", type=int, default=5,
                        help="Permutation count grows as num_fields!; keep this small.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1005])
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)
    print(f"{'seed':>6}{'orders':>8}{'worst':>9}{'median':>9}{'best':>9}"
          f"{'policy':>9}{'nearest':>9}{'gap':>8}{'pctile':>9}")
    for seed in args.seeds:
        environment = InstanceGenerator(config).generate(
            num_fields=args.num_fields, seed=seed, randomize_each_reset=False
        )
        field_ids = [field.field_id for field in environment.fields]
        scores = exhaustive_reference(environment, field_ids)
        if not scores:
            print(f"{seed:>6}  no feasible order")
            continue

        torch.manual_seed(0)
        actor = CoveragePolicy(environment, config=config)
        if args.checkpoint:
            agent = PPOAgent(actor, config)
            agent.load_checkpoint(args.checkpoint, map_location="cpu")
        actor.eval()
        policy = _policy_reward(actor, environment)
        nearest = _nearest_field_reward(environment)
        # Fraction of the 120 orders the policy scores better than. Higher is
        # better; ~50% would mean "no ordering strategy at all".
        beaten = sum(1 for value in scores if value < policy)
        percentile = 100.0 * beaten / len(scores)
        print(f"{seed:>6}{len(scores):>8}{scores[0]:>9.2f}"
              f"{scores[len(scores) // 2]:>9.2f}{scores[-1]:>9.2f}"
              f"{policy:>9.2f}{nearest:>9.2f}{scores[-1] - policy:>8.2f}"
              f"{percentile:>8.0f}%")


if __name__ == "__main__":
    main()
