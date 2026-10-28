"""What does the beam-search solution do differently from the two-line rule?

The beam oracle proved the rule is at least 5.6% off optimum. This decomposes both
solutions visit by visit so the improvement can be attributed to a decision:

* visit count / resupply count      -> the *structure* of the trip
* coverage vs transit distance       -> gates/strokes vs ordering
* hop statistics                     -> the shape of the tour
"""

from __future__ import annotations

import argparse
import math
import statistics


def _distance(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def decompose(environment, sequence) -> dict | None:
    """Replay a sequence and record how the distance splits."""
    environment.reset()
    transit = coverage = resupply = 0.0
    return_leg = 0.0
    resupplies = 0
    visits = []
    hops = []
    previous_end = environment.depot.get_position()
    for field_id, entry_id, exit_id in sequence:
        actions = environment.get_action_candidates()
        mask = environment.get_action_mask(actions)
        match = [
            index for index, action in enumerate(actions)
            if action.field_id == field_id
            and action.entry_boundary_id == entry_id
            and action.exit_boundary_id == exit_id
            and bool(mask[index])
        ]
        if not match:
            return None
        field = environment.get_field(field_id)
        entry = next(
            (b.center for b in field.boundaries if b.id == entry_id), None
        )
        _state, _reward, done, info = environment.step(actions[match[0]])
        path = [tuple(point) for point in info.get("path", [])]
        length = sum(_distance(a, b) for a, b in zip(path[:-1], path[1:])) if len(path) > 1 else 0.0
        moved = min(_distance(previous_end, entry), length) if entry else 0.0
        transit += moved
        coverage += max(0.0, length - moved)
        hops.append(moved)
        if info.get("final_return_distance"):
            return_leg += float(info["final_return_distance"])
        if info.get("auto_resupply"):
            resupplies += 1
            # Only the depot leg belongs to resupply; the executed coverage segment is
            # already counted in ``coverage`` above. Using the whole step length here
            # double-counted it and made the components exceed the mission total.
            resupply += float(info.get("resupply_or_return_distance",
                                       info.get("resupply_distance", 0.0)))
        visits.append(field_id)
        previous_end = path[-1] if path else previous_end
        if done:
            break
    summary = environment.get_summary()
    return {
        "final_return": float(return_leg),
        "visits": len(visits),
        "resupplies": resupplies,
        "distinct_parcels": len(set(visits)),
        "coverage": coverage,
        "transit": transit,
        "resupply_steps": resupply,
        "total": float(summary.get("total_distance", 0.0)),
        "mean_hop": statistics.mean(hops) if hops else 0.0,
        "max_hop": max(hops) if hops else 0.0,
        "sequence": visits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-count", type=int, default=5)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--branches", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--budget-seconds", type=float, default=400.0)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
    from coverage_env.env.instance_generator import InstanceGenerator
    from coverage_env.rl.evaluation.beam_oracle import beam_search
    from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
    from coverage_env.rl.evaluation.joint_search import trace

    config = load_config(DEFAULT_CONFIG_PATH)
    config["environment"]["num_fields"] = {"min": 3, "max": 20}

    header = (
        f"{'seed':>6}{'谁的':>5}{'访问':>5}{'补给':>5}{'覆盖m':>8}{'转场m':>8}"
        f"{'补给步m':>9}{'均跳m':>7}{'总m':>8}"
    )
    print(header)
    for index in range(args.instances):
        seed = args.base_seed + index
        environment = InstanceGenerator(config).generate(
            num_fields=args.field_count, seed=seed, randomize_each_reset=False
        )
        environment.reset()
        rule_sequence = trace(environment, _nearest_field_choice)
        rule = decompose(environment, rule_sequence) if rule_sequence else None
        best, _steps = beam_search(
            environment, beam=args.beam, branches=args.branches,
            max_depth=args.max_depth, budget_seconds=args.budget_seconds,
            verbose=False,
        )
        beam = decompose(environment, best[1]) if best else None
        for label, data in (("规则", rule), ("束搜", beam)):
            if data is None:
                continue
            print(
                f"{seed:>6}{label:>5}{data['visits']:>5}{data['resupplies']:>5}"
                f"{data['coverage']:>8.0f}{data['transit']:>8.0f}"
                f"{data['resupply_steps']:>9.0f}{data['mean_hop']:>7.0f}"
                f"{data['total']:>8.0f}"
            )
        if rule and beam:
            print(
                f"      差: 总 {beam['total']-rule['total']:+.0f} m "
                f"({(beam['total']-rule['total'])/rule['total']:+.1%}) | "
                f"覆盖 {beam['coverage']-rule['coverage']:+.0f} | "
                f"转场 {beam['transit']-rule['transit']:+.0f} | "
                f"访问 {beam['visits']-rule['visits']:+d}"
            )


if __name__ == "__main__":
    main()
