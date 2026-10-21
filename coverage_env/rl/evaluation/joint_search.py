"""Joint local search over (field order, entry/exit gates) -- the missing reference.

Why this exists
---------------
Every reference we had was partial. ``order_reference`` enumerates orders but fixes the
gate rule; the centroid-TSP reference is purely geometric; the oracle test forced a
geometric order but inherited the policy's gates (an out-of-distribution combination).
So we could never say how far the policy is from optimal.

This module searches the *joint* space with the real simulator as the objective:

* a candidate is a full trajectory -- a list of ``(field, entry, exit)`` triples,
* it is scored by replaying it through the environment (same reward, same resupply
  logic, same workload estimate as training),
* moves are 2-opt/or-opt style reorderings plus per-visit gate swaps,
* hill climbing from several starts, so we also see whether the starts converge.

A trajectory that becomes infeasible part-way, or that does not reach mission complete,
scores ``None`` and is never accepted.

Usage
-----
    PYTHONPATH=. python -m coverage_env.rl.evaluation.joint_search \
        --checkpoint coverage_env/results/run_long2000_v11/ckpt_update_1400.pt \
        --field-count 5 --instances 5 --sweeps 12
"""

from __future__ import annotations

import argparse
import statistics

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator

MAX_STEPS = 120


def replay(environment, sequence) -> float | None:
    """Score a trajectory with the real simulator; ``None`` if it is invalid."""
    environment.reset()
    total = 0.0
    done = False
    info: dict = {}
    for field_id, entry, exit_boundary in sequence:
        actions = environment.get_action_candidates()
        mask = environment.get_action_mask(actions)
        match = [
            index for index, action in enumerate(actions)
            if action.field_id == field_id
            and action.entry_boundary_id == entry
            and action.exit_boundary_id == exit_boundary
            and bool(mask[index])
        ]
        if not match:
            return None
        _state, reward, done, info = environment.step(actions[match[0]])
        total += float(reward)
        if done:
            break
    if not done or str(info.get("termination_reason", "")) != "mission_complete":
        return None
    return total


def trace(environment, choose) -> list:
    """Record the ``(field, entry, exit)`` triples a policy takes."""
    state = environment.reset()
    sequence, done = [], False
    for _ in range(MAX_STEPS):
        action = choose(environment, state)
        if action is None:
            break
        sequence.append(
            (action.field_id, action.entry_boundary_id, action.exit_boundary_id)
        )
        state, _reward, done, info = environment.step(action)
        if done:
            break
    if not done or str(info.get("termination_reason", "")) != "mission_complete":
        return []
    return sequence


def _variants(environment, field_id):
    """All feasible ``(entry, exit)`` pairs of a parcel for repair moves."""
    actions = environment.get_action_candidates()
    mask = environment.get_action_mask(actions)
    return [
        (action.entry_boundary_id, action.exit_boundary_id)
        for action, feasible in zip(actions, mask)
        if bool(feasible) and action.field_id == field_id
    ]


def local_search(environment, start, sweeps=12):
    """Hill climbing: reorder moves plus per-visit gate swaps."""
    best, best_score = list(start), replay(environment, start)
    if best_score is None or not best:
        return None, None
    for _sweep in range(sweeps):
        improved = False
        # --- gate swaps: for each visit, try every feasible pair of that parcel
        for position in range(len(best)):
            field_id = best[position][0]
            for entry, exit_boundary in _variants(environment, field_id):
                if (entry, exit_boundary) == best[position][1:]:
                    continue
                candidate = list(best)
                candidate[position] = (field_id, entry, exit_boundary)
                score = replay(environment, candidate)
                if score is not None and score > best_score + 1e-9:
                    best, best_score, improved = candidate, score, True
        # --- reorder moves: 2-opt reversal and pairwise swap
        count = len(best)
        for i in range(count):
            for j in range(i + 1, count):
                for candidate in (
                    best[:i] + best[i:j + 1][::-1] + best[j + 1:],
                    best[:i] + [best[j]] + best[i + 1:j] + [best[i]] + best[j + 1:],
                ):
                    score = replay(environment, candidate)
                    if score is not None and score > best_score + 1e-9:
                        best, best_score, improved = candidate, score, True
        if not improved:
            break
    return best, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--field-count", type=int, default=5)
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--sweeps", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()

    from coverage_env.rl.decoder.action_decoder import CoveragePolicy
    from coverage_env.rl.decoder.hierarchical_decoder import checkpoint_decoder_flags
    from coverage_env.rl.ppo.ppo import PPOAgent
    from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice

    config = load_config(DEFAULT_CONFIG_PATH)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu")
        config["rl"]["decoder"].update(checkpoint_decoder_flags(saved["actor"]))
    seed_environment = InstanceGenerator(config).generate(
        num_fields=args.field_count, seed=args.base_seed, randomize_each_reset=False
    )
    policy = CoveragePolicy(seed_environment, config=config)
    if args.checkpoint:
        PPOAgent(policy, config).load_checkpoint(args.checkpoint, map_location="cpu")
    policy.eval()

    def policy_choice(environment, state):
        with torch.no_grad():
            output = policy(state, environment=environment)
            distribution = output["action_distribution"]
            if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0:
                return None
            return policy.greedy_action(output)

    print(
        f"{'seed':>6}{'策略':>9}{'从策略爬':>10}{'启发式':>9}{'从启发式爬':>12}"
        f"{'局部最优':>10}{'策略差距':>10}"
    )
    gaps, policy_gaps = [], []
    for index in range(args.instances):
        seed = args.base_seed + args.field_count * 100 + index
        environment = InstanceGenerator(config).generate(
            num_fields=args.field_count, seed=seed, randomize_each_reset=False
        )
        policy_sequence = trace(environment, policy_choice)
        policy_score = replay(environment, policy_sequence) if policy_sequence else None
        heuristic_sequence = trace(environment, _nearest_field_choice)
        heuristic_score = replay(environment, heuristic_sequence) if heuristic_sequence else None

        _seq_a, from_policy = local_search(environment, policy_sequence, args.sweeps) if policy_sequence else (None, None)
        _seq_b, from_heuristic = local_search(environment, heuristic_sequence, args.sweeps) if heuristic_sequence else (None, None)
        candidates = [value for value in (from_policy, from_heuristic) if value is not None]
        best = max(candidates) if candidates else None
        if None in (policy_score, best):
            print(f"{seed:>6}  评估失败")
            continue
        gaps.append(best - policy_score)
        print(
            f"{seed:>6}{policy_score:>9.2f}{from_policy:>10.2f}"
            f"{heuristic_score:>9.2f}{from_heuristic:>12.2f}{best:>10.2f}{best - policy_score:>+10.2f}"
        )
    if gaps:
        error = 1.96 * statistics.stdev(gaps) / len(gaps) ** 0.5 if len(gaps) > 1 else 0.0
        print(f"\n策略距离（搜索到的最好 − 策略）: {statistics.mean(gaps):+.2f} ± {error:.2f} (95% CI)")
        print("距离越小说明策略越接近这个搜索能找到的最优；距离大说明还有空间")


if __name__ == "__main__":
    main()
