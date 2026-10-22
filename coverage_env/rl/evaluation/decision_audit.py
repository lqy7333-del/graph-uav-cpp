"""Separate the two decisions: how much cost each one can actually move.

The report that prompted this conflated "the in-parcel coverage cost is local" with
"it leaves no room". Those are different claims: the entry/exit choice *is* part of
the action, so the coverage distance it controls is controllable cost. Only a
measurement can say how much of it is left on the table.

This module audits the two decision types separately, holding the other fixed:

* ``gate_audit`` -- for one parcel visit, enumerate every legal ``(entry, exit)`` pair
  and record the driven length (transit + in-parcel path). Reports the spread across
  pairs and what the myopic "nearest entry" rule leaves versus the best pair.
The order dimension is audited by ``joint_search`` (reverse/swap/gate moves from the
rule's own solution), which is the same "strong baseline" question asked in reverse.


Usage:
    PYTHONPATH=. python -m coverage_env.rl.evaluation.decision_audit \
        --field-count 10 --instances 3
"""

from __future__ import annotations

import argparse
import math
import statistics

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator


def _distance(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _path_length(info, previous, entry_center) -> tuple[float, float]:
    path = [tuple(point) for point in info.get("path", [])]
    total = sum(_distance(a, b) for a, b in zip(path[:-1], path[1:]))
    transit = _distance(previous, entry_center) if entry_center else 0.0
    return total, min(transit, total)


def gate_audit(environment, field_ids) -> list[dict]:
    """For each parcel: driven length of every legal gate, from a fixed approach."""
    results = []
    for field_id in field_ids:
        environment.reset()
        position = environment.depot.get_position()
        actions = environment.get_action_candidates()
        mask = environment.get_action_mask(actions)
        lengths = []
        nearest_entry = None
        best_nearest = float("inf")
        field = environment.get_field(field_id)
        entries = {boundary.id: boundary.center for boundary in field.boundaries}
        for action, feasible in zip(actions, mask):
            if not feasible or action.field_id != field_id:
                continue
            entry = entries.get(action.entry_boundary_id)
            if entry is None:
                continue
            environment.reset()
            _state, _reward, _done, info = environment.step(action)
            total, _transit = _path_length(info, position, entry)
            if total <= 0.0:
                continue
            lengths.append(total)
            approach = _distance(position, entry)
            if approach < best_nearest:
                best_nearest = approach
                nearest_entry = total
        if len(lengths) >= 2 and nearest_entry is not None:
            results.append({
                "field": field_id,
                "pairs": len(lengths),
                "best": min(lengths),
                "worst": max(lengths),
                "rule": nearest_entry,
                "mean": statistics.mean(lengths),
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-count", type=int, default=10)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=9000)
    args = parser.parse_args()

    config = load_config(DEFAULT_CONFIG_PATH)
    print(
        f"{'seed':>6}{'访问数':>7}{'门组合/块':>10}{'单次均值m':>11}"
        f"{'最好-最差':>10}{'规则 vs 最优':>13}{'规则损失':>9}"
    )
    losses, spreads = [], []
    for index in range(args.instances):
        environment = InstanceGenerator(config).generate(
            num_fields=args.field_count, seed=args.base_seed + index,
            randomize_each_reset=False,
        )
        environment.reset()
        field_ids = [field.field_id for field in environment.fields]
        rows = gate_audit(environment, field_ids)
        if not rows:
            continue
        mean_length = statistics.mean(row["mean"] for row in rows)
        spread = statistics.mean((row["worst"] - row["best"]) / row["mean"] for row in rows)
        rule_gap = statistics.mean((row["rule"] - row["best"]) / row["mean"] for row in rows)
        losses.append(rule_gap)
        spreads.append(spread)
        print(
            f"{args.base_seed + index:>6}{len(rows):>7}"
            f"{statistics.mean(row['pairs'] for row in rows):>10.1f}"
            f"{mean_length:>11.0f}{spread:>10.1%}{rule_gap:>13.1%}{rule_gap:>9.1%}"
        )
    if losses:
        se = statistics.stdev(losses) / len(losses) ** 0.5 if len(losses) > 1 else 0.0
        print(
            f"\n同一块地不同门组合的成本差异（相对均值）: {statistics.mean(spreads):.1%}"
        )
        print(
            f"「最近入口」规则相对最优门的损失: {statistics.mean(losses):.1%} ± {1.96 * se:.1%}"
        )
        print(
            "=> 这是「只改门、不改顺序」能拿回的空间；若接近 0，说明入口/出口这一维已饱和"
        )


if __name__ == "__main__":
    main()
