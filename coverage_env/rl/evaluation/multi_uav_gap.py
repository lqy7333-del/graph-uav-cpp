"""Measure whether a multi-UAV makespan objective leaves room over a myopic rule.

A multi-UAV *sum-of-distance* objective decomposes per vehicle, so it adds no
headroom. A makespan objective (finish every parcel as early as possible) couples
the vehicles, but the natural myopic rule -- "whichever UAV is free first takes its
nearest unassigned parcel" -- is list scheduling, which is a decent approximation for
the *time* while ignoring the *geometry*. That mismatch is the suspected gap.

This module measures it directly, without training:

* ``route_length`` replays a set of parcels through the real simulator (same coverage
  geometry, same workload refinement) and returns the driven distance,
* the makespan of a partition is ``max_i distance_i / speed`` (UAVs run concurrently),
* a greedy list-scheduling partition is compared against a local search over the
  partition, so the number reported is exactly "how much better than the myopic rule
  can anything be".

Usage:
    PYTHONPATH=. python -m coverage_env.rl.evaluation.multi_uav_gap \
        --field-count 10 --instances 2 --search-steps 30
"""

from __future__ import annotations

import argparse
import math
import statistics

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator

MAX_STEPS = 200


def _distance(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _greedy_action(environment, allowed, position):
    """Cheapest gate of the nearest allowed parcel (the two-line rule)."""
    actions = environment.get_action_candidates()
    mask = environment.get_action_mask(actions)
    best_field, best_distance = None, float("inf")
    for action, feasible in zip(actions, mask):
        if not feasible or action.field_id not in allowed:
            continue
        field = environment.get_field(action.field_id)
        gap = _distance(position, field.centroid)
        if gap < best_distance:
            best_field, best_distance = action.field_id, gap
    if best_field is None:
        return None
    field = environment.get_field(best_field)
    entries = {boundary.id: boundary.center for boundary in field.boundaries}
    best, best_key = None, float("inf")
    for action, feasible in zip(actions, mask):
        if not feasible or action.field_id != best_field:
            continue
        entry = entries.get(action.entry_boundary_id)
        if entry is None:
            continue
        key = _distance(position, entry)
        if key < best_key:
            best, best_key = action, key
    return best


def route_length(environment, parcels: set) -> float:
    """Distance flown to cover exactly ``parcels``, starting and ending at the depot."""
    environment.reset()
    covered = set()
    total = 0.0
    for _ in range(MAX_STEPS):
        if covered >= parcels:
            break
        position = environment.uavs[0].current_position or environment.depot.get_position()
        action = _greedy_action(environment, parcels - covered, position)
        if action is None:
            break
        _state, _reward, _done, info = environment.step(action)
        path = [tuple(point) for point in info.get("path", [])]
        total += sum(_distance(a, b) for a, b in zip(path[:-1], path[1:]))
        field = environment.get_field(action.field_id)
        if field is not None and not field.get_uncovered_points():
            covered.add(action.field_id)
        elif info.get("success"):
            covered.add(action.field_id)
    # Return leg to the depot counts toward the makespan.
    last = environment.uavs[0].current_position
    if last is not None:
        total += _distance(last, environment.depot.get_position())
    return total


def makespan(environment, partitions, speed: float) -> float:
    return max(route_length(environment, set(group)) / speed for group in partitions)


def greedy_partition(environment, field_ids, vehicles, speed):
    """List scheduling: whichever UAV frees first takes its nearest parcel."""
    parcels = {field_id: environment.get_field(field_id).centroid for field_id in field_ids}
    groups = [[] for _ in range(vehicles)]
    finish = [0.0] * vehicles
    position = [environment.depot.get_position() for _ in range(vehicles)]
    remaining = dict(parcels)
    while remaining:
        index = min(range(vehicles), key=lambda k: finish[k])
        target = min(remaining, key=lambda fid: _distance(position[index], remaining[fid]))
        gap = _distance(position[index], remaining[target])
        groups[index].append(target)
        finish[index] += gap / speed
        position[index] = remaining[target]
        remaining.pop(target)
    return groups


def search_partition(environment, groups, speed, steps):
    """Move single parcels between vehicles while the makespan improves."""
    best = [list(group) for group in groups]
    best_cost = makespan(environment, best, speed)
    for _step in range(steps):
        improved = False
        loads = [route_length(environment, set(group)) / speed for group in best]
        heavy = max(range(len(best)), key=lambda k: loads[k])
        for light in range(len(best)):
            if light == heavy:
                continue
            for parcel in list(best[heavy]):
                if len(best[heavy]) <= 1:
                    break
                candidate = [list(group) for group in best]
                candidate[heavy].remove(parcel)
                candidate[light].append(parcel)
                cost = makespan(environment, candidate, speed)
                if cost < best_cost - 1e-9:
                    best, best_cost, improved = candidate, cost, True
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_cost


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-count", type=int, default=10)
    parser.add_argument("--instances", type=int, default=2)
    parser.add_argument("--vehicles", type=int, default=2)
    parser.add_argument("--search-steps", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=9000)
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)
    print(
        f"{'seed':>6}{'单机耗时s':>12}{'贪心makespan':>14}{'搜索makespan':>14}{'差距':>9}"
    )
    gaps = []
    for index in range(args.instances):
        environment = InstanceGenerator(config).generate(
            num_fields=args.field_count, seed=args.base_seed + index,
            randomize_each_reset=False,
        )
        environment.reset()
        speed = float(environment.uavs[0].speed)
        field_ids = [field.field_id for field in environment.fields]

        single = route_length(environment, set(field_ids)) / speed
        groups = greedy_partition(environment, field_ids, args.vehicles, speed)
        greedy_cost = makespan(environment, groups, speed)
        _best, search_cost = search_partition(
            environment, groups, speed, args.search_steps
        )
        gaps.append((greedy_cost - search_cost) / greedy_cost)
        print(
            f"{args.base_seed + index:>6}{single:>12.0f}{greedy_cost:>14.0f}"
            f"{search_cost:>14.0f}{(greedy_cost - search_cost) / greedy_cost:>8.1%}"
        )
    if gaps:
        print(
            f"\n贪心规则相对搜索最优的平均差距: {statistics.mean(gaps):.1%}"
        )
        print("差距 <5% 说明近视规则已够用；>15% 才值得为这个目标训练策略")


if __name__ == "__main__":
    main()
