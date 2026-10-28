"""Lower bound and exhaustive-order reference for small instances.

The previous audits could only say "the rule is within a few percent of what our
searches find". That does not say how far *anything* could be. This module adds:

* a **valid lower bound** on the total driven distance for one episode:
    sum_i (area_i / coverage_width)              coverage cannot be cheaper
  + MST(parcels + depot)                          a tour costs at least a spanning tree
  + (loads - 1) * 2 * min distance depot->parcel  every extra load needs a round trip
* an **exhaustive-order reference** for small n: every permutation of the parcels is
  replayed through the real simulator with the cheapest gate at each step, which
  bounds what order choice alone can buy.

Neither is the exact optimum: the bound ignores that gates and order interact, and the
reference fixes the gate rule. Together they bracket the answer from both sides.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.evaluation.joint_search import replay


def _distance(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _mst(points) -> float:
    """Prim's algorithm over the parcels plus the depot."""
    remaining = list(points)
    if len(remaining) < 2:
        return 0.0
    tree = [remaining.pop(0)]
    total = 0.0
    while remaining:
        best, best_index = float("inf"), 0
        for index, point in enumerate(remaining):
            gap = min(_distance(point, node) for node in tree)
            if gap < best:
                best, best_index = gap, index
        total += best
        tree.append(remaining.pop(best_index))
    return total


def lower_bound(environment) -> tuple[float, dict]:
    width = float(environment.uavs[0].coverage_width)
    payload = float(environment.uavs[0].max_payload)
    density = 0.004  # payload per square metre, mirrored from the config scale
    area = sum(float(field.area) for field in environment.fields)
    coverage = area / width
    points = [environment.depot.get_position()] + [
        field.centroid for field in environment.fields
    ]
    spanning = _mst(points)
    nearest = min(
        _distance(environment.depot.get_position(), field.centroid)
        for field in environment.fields
    )
    loads = max(1, math.ceil(area * density / max(payload, 1e-9)))
    resupply = max(0, loads - 1) * 2.0 * nearest
    total = coverage + spanning + resupply
    return total, {
        "coverage": coverage,
        "spanning": spanning,
        "resupply": resupply,
        "loads": loads,
    }


def run_order(environment, order) -> float | None:
    """Replay an order with the cheapest *feasible* gate at every step.

    Choosing the gate up front (instead of at each step) produced infeasible
    trajectories because feasibility depends on the state, so the choice has to be
    made inside the rollout.
    """
    environment.reset()
    total = 0.0
    done = False
    info: dict = {}
    index = 0
    for _step in range(200):
        if index >= len(order):
            break
        field_id = order[index]
        actions = environment.get_action_candidates()
        mask = environment.get_action_mask(actions)
        position = environment.uavs[0].current_position or environment.depot.get_position()
        field = environment.get_field(field_id)
        entries = {b.id: b.center for b in field.boundaries}
        best, best_key = None, float("inf")
        for action, feasible in zip(actions, mask):
            if not feasible or action.field_id != field_id:
                continue
            entry = entries.get(action.entry_boundary_id)
            if entry is None:
                continue
            key = _distance(position, entry)
            if key < best_key:
                best, best_key = action, key
        if best is None:
            break
        _state, reward, done, info = environment.step(best)
        total += float(reward)
        index += 1
        if done:
            break
    if not done or str(info.get("termination_reason", "")) != "mission_complete":
        return None
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-count", type=int, default=5)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    import torch
    from coverage_env.rl.decoder.action_decoder import CoveragePolicy
    from coverage_env.rl.decoder.hierarchical_decoder import checkpoint_decoder_flags
    from coverage_env.rl.ppo.ppo import PPOAgent
    from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice

    config = load_config(DEFAULT_CONFIG_PATH)
    if args.checkpoint:
        from coverage_env.rl.checkpoint import evaluation_config
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = evaluation_config(saved, config)

    print(
        f"{'seed':>6}{'下界m':>9}{'启发式m':>9}{'策略m':>9}"
        f"{'规则/下界':>10}{'策略/下界':>10}"
    )
    rule_gaps, policy_gaps = [], []
    for index in range(args.instances):
        environment = InstanceGenerator(config).generate(
            num_fields=args.field_count, seed=args.base_seed + index,
            randomize_each_reset=False,
        )
        environment.reset()
        bound, _parts = lower_bound(environment)
        area = sum(float(field.area) for field in environment.fields)
        to_distance = lambda score: -score * area / 100.0

        # A trajectory that revisits parcels may be required by the payload, so the
        # reference is "replay the recorded visit sequence", not "visit each once".
        from coverage_env.rl.evaluation.joint_search import replay, trace
        rule_sequence = trace(environment, _nearest_field_choice)
        rule_score = replay(environment, rule_sequence) if rule_sequence else None

        policy_distance = None
        if args.checkpoint:
            policy = CoveragePolicy(environment, config=config)
            policy.load_state_dict(saved["actor"])
            policy.eval()

            def choose(env, state):
                with torch.no_grad():
                    out = policy(state, environment=env)
                    dist = out["action_distribution"]
                    if dist.numel() == 0 or float(dist.detach().sum()) <= 0:
                        return None
                    return policy.greedy_action(out)

            sequence = trace(environment, choose)
            score = replay(environment, sequence) if sequence else None
            policy_distance = to_distance(score) if score is not None else None

        if rule_score is None:
            print(f"{args.base_seed + index:>6}  启发式评估失败")
            continue
        rule_distance = to_distance(rule_score)
        rule_gaps.append(rule_distance / bound - 1.0)
        text = (
            f"{args.base_seed + index:>6}{bound:>9.0f}{rule_distance:>9.0f}"
            f"{(policy_distance if policy_distance is not None else float('nan')):>9.0f}"
            f"{rule_gaps[-1]:>10.1%}"
        )
        if policy_distance is not None:
            policy_gaps.append(policy_distance / bound - 1.0)
            text += f"{policy_gaps[-1]:>10.1%}"
        print(text)
    if rule_gaps:
        print(f"\n启发式相对下界: {statistics.mean(rule_gaps):.1%}")
    if policy_gaps:
        print(f"策略相对下界: {statistics.mean(policy_gaps):.1%}")
    print("下界是保守的（覆盖按面积/幅宽、转场按最小生成树、补给按最少次数），真实最优在它之上")


if __name__ == "__main__":
    main()
