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


def _branch_action_context(environment, limit, *, diverse_fields=False):
    """Return selected branches and the complete candidate set/mask."""
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

    if diverse_fields:
        # Spread the first branches across fields before spending remaining
        # budget on alternative gates. The old nearest-first cap could spend all
        # branches on one nearby field, leaving no global field comparison.
        by_field = {}
        for distance, action in rows:
            by_field.setdefault(action.field_id, []).append((distance, action))
        groups = sorted(
            by_field.values(), key=lambda group: group[0][0]
        )
        if len(groups) > limit and limit > 1:
            # When not every parcel fits, sample evenly over the distance-sorted
            # field list so distant candidates are not systematically omitted.
            group_indices = {
                round(index * (len(groups) - 1) / (limit - 1))
                for index in range(limit)
            }
            selected_groups = [
                group for index, group in enumerate(groups)
                if index in group_indices
            ]
        else:
            selected_groups = groups[:limit]
        selected = [group[0][1] for group in selected_groups]
        round_index = 1
        while len(selected) < limit:
            added = False
            for group in selected_groups:
                if round_index < len(group) and len(selected) < limit:
                    selected.append(group[round_index][1])
                    added = True
            if not added:
                break
            round_index += 1
    else:
        selected = [action for _distance_value, action in rows[:limit]]

    return selected, actions, mask


def branch_actions(environment, limit, *, diverse_fields=False):
    """Feasible actions, nearest-entry first and capped at ``limit``.

    ``diverse_fields=True`` distributes first branches over the distance-sorted
    fields (evenly across the list when fields outnumber branches), then adds
    alternative entry/exit pairs. This is useful for global action labels.
    """
    selected, _actions, _mask = _branch_action_context(
        environment, limit, diverse_fields=diverse_fields
    )
    return selected


def select_beam(frontier, beam, *, diverse_frontier=False):
    """Pick the surviving nodes, optionally keeping one leader per root field.

    ``diverse_frontier`` reserves a slot for the best node of every root action's
    field before filling the rest by accumulated reward. Without it the first-layer
    beam can be filled by several gates of one parcel, so every completed path
    shares a root field and the cost-to-go labels never compare different parcels.
    Nodes are ``(value, sequence, state[, trace])`` and ``sequence[0][0]`` is the
    root action's field id.
    """
    ordered = sorted(frontier, key=lambda item: item[0], reverse=True)
    if not diverse_frontier:
        return ordered[:beam]
    leaders = {}
    others = []
    for item in ordered:
        root_field = item[1][0][0] if item[1] else None
        if root_field not in leaders:
            leaders[root_field] = item
        else:
            others.append(item)
    leading = sorted(leaders.values(), key=lambda item: item[0], reverse=True)
    return (leading + others)[:beam]


def beam_search(
    environment, *, beam, branches, max_depth, budget_seconds,
    verbose=True, reset=True, return_trace=False,
    trace_context_builder=None, diverse_fields=False,
    diverse_frontier=False, max_completed_paths=4096,
):
    import torch  # noqa: F401  (the env pulls torch in through its own imports)

    root = copy.deepcopy(environment)
    # Full-episode oracle calls start from reset by default. Cost-to-go queries
    # pass reset=False so every candidate is evaluated from the exact same
    # intermediate UAV/coverage/resource state.
    if reset:
        root.reset()
    if return_trace and trace_context_builder is None:
        raise ValueError("return_trace requires trace_context_builder.")
    if max_completed_paths <= 0:
        raise ValueError("max_completed_paths must be positive.")
    nodes = [(0.0, [], root, [])]
    best = None
    trace_contexts = []
    completed_paths = []
    dropped_completed_paths = 0
    started = time.time()
    steps = 0
    termination_reason = "max_depth"
    for depth in range(max_depth):
        frontier = []
        budget_exhausted = False
        for total, sequence, state, trace_path in nodes:
            if state.uavs and state.get_summary().get("coverage_ratio") is not None:
                pass
            if return_trace:
                actions, all_actions, mask = _branch_action_context(
                    state, branches, diverse_fields=diverse_fields
                )
                context_id = len(trace_contexts)
                context = trace_context_builder(state, all_actions, mask)
                context["costs"] = [None] * len(all_actions)
                context["statuses"] = [
                    "infeasible" if not feasible else "branch_pruned"
                    for feasible in mask
                ]
                context["search_steps"] = [0] * len(all_actions)
                context["completed_path_count"] = [0] * len(all_actions)
                trace_contexts.append(context)
                action_indices = {action: index for index, action in enumerate(all_actions)}
            else:
                actions = branch_actions(state, branches)
                all_actions = mask = None
                context_id = None
            distance_before = float(state.get_summary()["total_distance"])
            for action in actions:
                if time.time() - started > budget_seconds:
                    budget_exhausted = True
                    break
                trial = copy.deepcopy(state)
                triple = (action.field_id, action.entry_boundary_id,
                          action.exit_boundary_id)
                try:
                    _s, reward, done, _info = trial.step(action)
                except Exception:
                    if return_trace:
                        context["statuses"][action_indices[action]] = "invalid_transition"
                    continue
                steps += 1
                if return_trace:
                    action_index = action_indices[action]
                    context["search_steps"][action_index] += 1
                    context["statuses"][action_index] = "expanded_no_complete_suffix"
                if _info.get("success") is False:
                    if return_trace:
                        context["statuses"][action_index] = "invalid_transition"
                    continue
                value = total + float(reward)
                candidate = (value, sequence + [triple], trial)
                if return_trace:
                    action_index = action_indices[action]
                    child_trace = trace_path + [
                        (context_id, action_index, distance_before)
                    ]
                else:
                    child_trace = []
                if done:
                    reason = str(_info.get("termination_reason", ""))
                    if reason == "mission_complete":
                        if best is None or value > best[0]:
                            best = candidate
                        if return_trace:
                            if len(completed_paths) < max_completed_paths:
                                completed_paths.append((trial, child_trace))
                            else:
                                dropped_completed_paths += 1
                else:
                    frontier.append((*candidate, child_trace))
            if budget_exhausted:
                break
        if budget_exhausted or time.time() - started > budget_seconds:
            termination_reason = "time_budget"
            if verbose:
                print(f"  [budget] stopped at depth {depth}, {steps} steps")
            break
        if not frontier:
            termination_reason = "frontier_empty"
            break
        nodes = select_beam(frontier, beam, diverse_frontier=diverse_frontier)
        if verbose:
            print(f"  depth {depth + 1}: {len(frontier)} nodes, best so far "
                  f"{nodes[0][0]:.2f}, steps {steps}, "
                  f"{time.time() - started:.0f}s")
    if not return_trace:
        return best, steps

    for final_environment, trace_path in completed_paths:
        final_distance = float(final_environment.get_summary()["total_distance"])
        for context_id, action_index, distance_before in trace_path:
            context = trace_contexts[context_id]
            cost = max(0.0, final_distance - distance_before)
            previous = context["costs"][action_index]
            context["costs"][action_index] = (
                cost if previous is None else min(float(previous), cost)
            )
            context["statuses"][action_index] = "completed_suffix_found"
            context["completed_path_count"][action_index] += 1

    pair_count = 0
    for context in trace_contexts:
        labeled = [
            cost for feasible, cost in zip(context["feasible_mask"], context["costs"])
            if feasible and cost is not None
        ]
        pair_count += sum(
            left != right
            for index, left in enumerate(labeled)
            for right in labeled[index + 1:]
        )

    trace_result = {
        "contexts": trace_contexts,
        "completed_paths": len(completed_paths),
        "dropped_completed_paths": dropped_completed_paths,
        "pair_count": pair_count,
        "termination_reason": termination_reason,
        "elapsed_seconds": time.time() - started,
        "expanded_steps": steps,
    }
    return best, steps, trace_result


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
