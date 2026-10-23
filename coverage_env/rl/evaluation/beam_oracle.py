"""Beam search whose cost oracle is the real environment.

A MILP is not faithful here: both the workload fraction and the in-parcel coverage path
are state-dependent geometric functions (`estimate_workload` keys on the full uncovered
grid, `boundary_aware_sweep` optimises orientation and stroke order per gate). A
discretised model therefore stops matching the simulator -- an earlier attempt reported
41% headroom and then failed to replay.

So this searches the *action sequences* with the simulator itself as the oracle: every
node is a branch of a deep-copied environment, every extension is a real ``step``. Any
solution found is therefore feasible by construction, which is exactly the property the
previous model lacked.

What it can establish:
* if it finds a sequence better than the rule, the rule is **provably not optimal** and
  the improvement is a **lower bound** on the true gap,
* if it finds nothing, no conclusion follows -- it only means this search could not
  reach a better solution.
"""

from __future__ import annotations

import argparse
import copy
import math
import statistics
import time


def _distance(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def branch_actions(environment, limit):
    """Feasible (parcel, gate) actions, nearest-entry first, capped at ``limit``."""
    actions = environment.get_action_candidates()
    mask = environment.get_action_mask(actions)
    position = environment.uavs[0].current_position or environment.depot.get_position()
    rows = []
    for action, feasible in zip(actions, mask):
        if not feasible:
            continue
        field = environment.get_field(action.field_id)
        entries = {boundary.id: boundary.center for boundary in field.boundaries}
        entry = entries.get(action.entry_boundary_id)
        if entry is None:
            continue
        rows.append((_distance(position, entry), action))
    rows.sort(key=lambda pair: pair[0])
    seen, picked = set(), []
    for _gap, action in rows:
        key = (action.field_id, action.entry_boundary_id, action.exit_boundary_id)
        if key in seen:
            continue
        seen.add(key)
        picked.append(action)
        if len(picked) >= limit:
            break
    return picked


def beam_search(
    environment, *, beam, branches, max_depth, budget_seconds,
    verbose=True, reset=True,
):
    import torch  # noqa: F401  (the env pulls torch in through its own imports)

    root = copy.deepcopy(environment)
    # Full-episode oracle calls start from reset by default. Cost-to-go queries
    # pass reset=False so every candidate is evaluated from the exact same
    # intermediate UAV/coverage/resource state.
    if reset:
        root.reset()
    nodes = [(0.0, [], root)]
    best = None
    started = time.time()
    steps = 0
    for depth in range(max_depth):
        frontier = []
        for total, sequence, state in nodes:
            if state.uavs and state.get_summary().get("coverage_ratio") is not None:
                pass
            actions = branch_actions(state, branches)
            for action in actions:
                if time.time() - started > budget_seconds:
                    break
                trial = copy.deepcopy(state)
                triple = (action.field_id, action.entry_boundary_id,
                          action.exit_boundary_id)
                try:
                    _s, reward, done, _info = trial.step(action)
                except Exception:
                    continue
                steps += 1
                value = total + float(reward)
                candidate = (value, sequence + [triple], trial)
                if done:
                    reason = str(_info.get("termination_reason", ""))
                    if reason == "mission_complete":
                        if best is None or value > best[0]:
                            best = candidate
                else:
                    frontier.append(candidate)
            if time.time() - started > budget_seconds:
                break
        if time.time() - started > budget_seconds:
            if verbose:
                print(f"  [budget] stopped at depth {depth}, {steps} steps")
            break
        if not frontier:
            break
        frontier.sort(key=lambda item: item[0], reverse=True)
        nodes = frontier[:beam]
        if verbose:
            print(f"  depth {depth + 1}: {len(frontier)} nodes, best so far "
                  f"{nodes[0][0]:.2f}, steps {steps}, "
                  f"{time.time() - started:.0f}s")
    return best, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-count", type=int, default=5)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--branches", type=int, default=12)
    parser.add_argument("--max-depth", type=int, default=14)
    parser.add_argument("--budget-seconds", type=float, default=1800.0)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
    from coverage_env.env.instance_generator import InstanceGenerator
    from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice
    from coverage_env.rl.evaluation.joint_search import replay, trace

    config = load_config(DEFAULT_CONFIG_PATH)
    config["environment"]["num_fields"] = {"min": 3, "max": 20}
    for index in range(args.instances):
        seed = args.base_seed + index
        environment = InstanceGenerator(config).generate(
            num_fields=args.field_count, seed=seed, randomize_each_reset=False
        )
        environment.reset()
        area = sum(float(field.area) for field in environment.fields)
        to_distance = lambda score: -score * area / 100.0

        rule_sequence = trace(environment, _nearest_field_choice)
        rule_score = replay(environment, rule_sequence) if rule_sequence else None
        if rule_score is None:
            print(f"seed {seed}: 规则评估失败")
            continue
        rule_distance = to_distance(rule_score)

        print(f"seed {seed}: 规则 {rule_distance:.0f} m，开始束搜索")
        best, steps = beam_search(
            environment, beam=args.beam, branches=args.branches,
            max_depth=args.max_depth, budget_seconds=args.budget_seconds,
        )
        if best is None:
            print(f"seed {seed}: 未找到更好的可行解（{steps} 步）")
            continue
        best_distance = to_distance(best[0])
        print(f"seed {seed}: 搜索最优 {best_distance:.0f} m "
              f"（相对规则 {(best_distance - rule_distance) / rule_distance:+.1%}），"
              f"访问 {len(best[1])} 次，{steps} 步评估")


if __name__ == "__main__":
    main()
