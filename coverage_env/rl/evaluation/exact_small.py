"""Exact solution for n=3 (and n=4) under an additive gate/order model.

Why a model instead of brute-force replay: every (order, gate) combination replayed
through the environment costs seconds and there are ~10^4 of them. But the cost of a
visit splits cleanly:

    in-parcel cost(parcel, entry, exit)   depends only on the two edges
    transit(exit_i -> entry_j)            depends only on the two points

so for a fixed order the gate choice is a short chain, solvable exactly by dynamic
programming over ``(visit index, gate)``. Enumerating the n! orders then gives the
exact optimum of this model.

The model is verified afterwards by replaying its best solution in the real simulator:
if the replayed distance matches the model, the decomposition holds.
"""

from __future__ import annotations

import argparse
import itertools
import math

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.planner.boundary_aware_sweep import BoundaryAwareSweepPlanner


def _distance(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _internal_cost(plan) -> float:
    """Distance flown inside the parcel: the strokes plus the scan-line transits."""
    total = 0.0
    for start, end in plan.coverage_strokes:
        total += _distance(start, end)
    for path in plan.scan_transit_paths:
        total += sum(_distance(a, b) for a, b in zip(path[:-1], path[1:]))
    return total


def gate_table(environment, field, planner, spacing):
    """Every legal gate of one parcel: internal cost and the two boundary points."""
    region = list(field.grid_points)
    table = {}
    for entry in field.boundaries:
        if entry.ring_id != 0:
            continue
        for exit_edge in field.boundaries:
            if exit_edge.ring_id != 0 or exit_edge.id == entry.id:
                continue
            try:
                plan = planner.plan(
                    region, field, entry, exit_edge,
                    current_position=environment.depot.get_position(),
                    lane_spacing=spacing,
                )
            except Exception:
                continue
            table[(entry.id, exit_edge.id)] = {
                "internal": _internal_cost(plan),
                "entry_point": plan.entry_point,
                "exit_point": plan.exit_point,
            }
    return table


def solve(environment, tables, field_ids):
    """Exact minimum over all orders and gate chains (additive model)."""
    depot = environment.depot.get_position()
    best = None
    for order in itertools.permutations(field_ids):
        # dp[(gate)] = (cost, chain) after the parcels placed so far.
        first = order[0]
        dp = {}
        for gate, data in tables[first].items():
            dp[gate] = (data["internal"] + _distance(depot, data["entry_point"]),
                        [gate])
        for field_id in order[1:]:
            nxt = {}
            for gate, data in tables[field_id].items():
                best_prev = None
                for prev_gate, (cost, chain) in dp.items():
                    prev_data = tables[order[order.index(field_id) - 1]][prev_gate]
                    value = cost + _distance(prev_data["exit_point"], data["entry_point"])
                    if best_prev is None or value < best_prev[0]:
                        best_prev = (value, chain + [gate])
                if best_prev is None:
                    continue
                nxt[gate] = (
                    best_prev[0] + data["internal"],
                    best_prev[1],
                )
            dp = nxt
        for gate, (cost, chain) in dp.items():
            last = order[-1]
            total = cost + _distance(tables[last][gate]["exit_point"], depot)
            if best is None or total < best[0]:
                best = (total, order, chain)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-count", type=int, default=3)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
    from coverage_env.rl.evaluation.joint_search import replay, trace

    config = load_config(DEFAULT_CONFIG_PATH)
    # n=3 is below the training band (5-20); widen the range for this audit only.
    config["environment"]["num_fields"] = {"min": 3, "max": 20}
    planner = BoundaryAwareSweepPlanner()
    print(f"{'seed':>6}{'模型最优m':>11}{'规则m':>9}{'规则超出':>9}{'验算m':>9}{'模型误差':>9}")
    gaps = []
    for index in range(args.instances):
        environment = InstanceGenerator(config).generate(
            num_fields=args.field_count, seed=args.base_seed + index,
            randomize_each_reset=False,
        )
        environment.reset()
        spacing = float(environment.uavs[0].coverage_width)
        field_ids = [field.field_id for field in environment.fields]
        tables = {
            field_id: gate_table(environment, environment.get_field(field_id),
                                 planner, spacing)
            for field_id in field_ids
        }
        best = solve(environment, tables, field_ids)
        if best is None:
            print(f"{args.base_seed + index:>6}  模型求解失败")
            continue
        model_cost, order, chain = best

        # Replay the model's solution in the real simulator to check the decomposition.
        sequence = [(field_id, gate[0], gate[1]) for field_id, gate in zip(order, chain)]
        environment.reset()
        score = replay(environment, sequence)
        area = sum(float(field.area) for field in environment.fields)
        replayed = -score * area / 100.0 if score is not None else float("nan")

        rule_sequence = trace(environment, _nearest_field_choice)
        rule_score = replay(environment, rule_sequence) if rule_sequence else None
        if rule_score is None:
            print(f"{args.base_seed + index:>6}  规则评估失败")
            continue
        rule_distance = -rule_score * area / 100.0
        gap = rule_distance / model_cost - 1.0
        gaps.append(gap)
        error = (replayed - model_cost) / model_cost if model_cost else float("nan")
        print(
            f"{args.base_seed + index:>6}{model_cost:>11.0f}{rule_distance:>9.0f}"
            f"{gap:>9.1%}{replayed:>9.0f}{error:>9.1%}"
        )
    if gaps:
        print(f"\n规则相对精确最优平均超出: {sum(gaps)/len(gaps):.1%}")
        print("模型误差是「模型最优」与「真实仿真重放」的差；接近 0 说明加法分解成立")


if __name__ == "__main__":
    main()
