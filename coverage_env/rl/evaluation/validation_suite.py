"""Fixed validation set with a cost decomposition, separated from training.

Why this exists
---------------
The training curve mixes two things: how the policy changed, and which instances the
random resampler happened to draw. Measured: across instances the reward standard
deviation is 2.69 while a single update averages only ~8.4 episodes, so the logged
mean has a standard error near 1.0 -- larger than the whole improvement of a long
run. Reading policy progress off that curve is not possible.

This harness instead evaluates a checkpoint on a **fixed** set of instances and
reports, per field count:

* total flight distance, and the split into transit / in-field coverage,
* the distance flown in steps where an automatic resupply happened and the resupply
  count (the resupply legs are the cost source that survives area normalisation),
* coverage ratio, completion and failure rate,
* the paired difference against a fixed nearest-field baseline with a confidence
  interval, so "this map is simply harder" cannot masquerade as progress.

Every episode uses the environment's own reward and termination, so the numbers are
directly comparable with training, but the instance set never changes.

Usage
-----
    PYTHONPATH=. python -m coverage_env.rl.evaluation.validation_suite \
        --checkpoint coverage_env/results/run_x/ckpt_update_800.pt
"""

from __future__ import annotations

import argparse
import statistics

import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.decoder.hierarchical_decoder import checkpoint_decoder_flags
from coverage_env.rl.ppo.ppo import PPOAgent

FIELD_COUNTS = (5, 10, 20)
INSTANCES_PER_COUNT = 5
BASE_SEED = 1000
MAX_STEPS = 120


def _distance(first, second) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _run(environment, choose) -> dict:
    """Execute one episode with ``choose(environment, state)``; return cost sources."""
    state = environment.reset()
    transit = coverage = resupply_distance = 0.0
    resupplies = 0
    reward = 0.0
    done = False
    info = {}
    for _ in range(MAX_STEPS):
        action = choose(environment, state)
        if action is None:
            break
        previous = environment.uavs[0].current_position
        state, step_reward, done, info = environment.step(action)
        reward += float(step_reward)
        path = [tuple(point) for point in info.get("path", [])]
        length = sum(
            _distance(a, b) for a, b in zip(path[:-1], path[1:])
        ) if len(path) > 1 else 0.0
        field = environment.get_field(info.get("selected_field"))
        entry = None
        if field is not None:
            for boundary in field.boundaries:
                if boundary.id == info.get("entry_boundary"):
                    entry = boundary.center
                    break
        moved = _distance(previous, entry) if (previous and entry) else 0.0
        moved = min(moved, length) if length > 0 else 0.0
        transit += moved
        coverage += max(0.0, length - moved)
        if info.get("auto_resupply"):
            resupplies += 1
            resupply_distance += length
        if done:
            break
    summary = environment.get_summary()
    return {
        "reward": reward,
        "total_distance": float(summary.get("total_distance", 0.0)),
        "transit": transit,
        "coverage": coverage,
        "resupply_steps_distance": resupply_distance,
        "resupplies": int(summary.get("resupply_count", resupplies)),
        "coverage_ratio": float(summary.get("coverage_ratio", 0.0)),
        "completed": bool(info.get("success")) and str(
            info.get("termination_reason", "")
        ) == "mission_complete",
        "reason": str(info.get("termination_reason", "?")),
        "area": float(environment.total_area),
    }


def _nearest_field_choice(environment, _state):
    """Two-level baseline: nearest unfinished parcel, then its cheapest gate.

    The gate part matters. Candidates of one parcel all share the same centroid
    distance, so ranking candidates by that distance and keeping the first feasible
    one silently picks whatever pair the enumerator emitted first (usually ``B0/B0``)
    -- an arbitrary gate that systematically depresses this baseline and inflates any
    "beats the baseline" claim. Greedy gate choice here is the one that minimises the
    approach distance to the entry boundary, which is what a myopic rule can know.
    """
    actions = environment.get_action_candidates()
    mask = environment.get_action_mask(actions)
    position = environment.uavs[0].current_position
    nearest_field, nearest_distance = None, float("inf")
    for action, feasible in zip(actions, mask):
        if not feasible:
            continue
        distance = _distance(position, environment.get_field(action.field_id).centroid)
        if distance < nearest_distance:
            nearest_field, nearest_distance = action.field_id, distance
    if nearest_field is None:
        return None
    field = environment.get_field(nearest_field)
    entries = {boundary.id: boundary.center for boundary in field.boundaries}
    best, best_key = None, float("inf")
    for action, feasible in zip(actions, mask):
        if not feasible or action.field_id != nearest_field:
            continue
        entry = entries.get(action.entry_boundary_id)
        if entry is None:
            continue
        key = _distance(position, entry)
        if key < best_key:
            best, best_key = action, key
    return best


def _policy_choice(policy):
    def choose(environment, state):
        with torch.no_grad():
            output = policy(state, environment=environment)
            distribution = output["action_distribution"]
            if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0:
                return None
            return CoveragePolicy.greedy_action(output)
    return choose


def _mean(values):
    return statistics.mean(values) if values else 0.0


def _ci(values):
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / (len(values) ** 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, nargs="*", default=[None])
    parser.add_argument("--instances", type=int, default=INSTANCES_PER_COUNT,
                        help="Instances per field count; more shrinks the paired CI.")
    args = parser.parse_args()

    for checkpoint in args.checkpoint:
        config = load_config(DEFAULT_CONFIG_PATH)
        if checkpoint:
            saved = torch.load(checkpoint, map_location="cpu")
            config["rl"]["decoder"].update(checkpoint_decoder_flags(saved["actor"]))
        torch.manual_seed(0)
        seed_environment = InstanceGenerator(config).generate(num_fields=5, seed=BASE_SEED)
        policy = CoveragePolicy(seed_environment, config=config)
        if checkpoint:
            PPOAgent(policy, config).load_checkpoint(checkpoint, map_location="cpu")
        policy.eval()
        name = (checkpoint or "untrained").split("/")[-1]
        if checkpoint and "/" in checkpoint:
            name = f"{checkpoint.split('/')[-2]}/{name}"
        print(f"=== {name} ===")
        header = (
            f"{'n':>4}{'reward':>9}{'dist':>8}{'transit':>9}{'coverage':>10}"
            f"{'补/距离':>10}{'覆盖':>8}{'完成':>7}{'配对(策略-NN)':>22}"
        )
        print(header)
        count_instances = int(args.instances)
        for count in FIELD_COUNTS:
            policy_rows, baseline_rows, deltas = [], [], []
            for index in range(count_instances):
                environment = InstanceGenerator(config).generate(
                    num_fields=count, seed=BASE_SEED + count * 100 + index,
                    randomize_each_reset=False,
                )
                policy_result = _run(environment, _policy_choice(policy))
                baseline = _run(environment, _nearest_field_choice)
                policy_rows.append(policy_result)
                baseline_rows.append(baseline)
                deltas.append(policy_result["reward"] - baseline["reward"])
            print(
                f"{count:>4}{_mean([r['reward'] for r in policy_rows]):>9.2f}"
                f"{_mean([r['total_distance'] for r in policy_rows]):>8.0f}"
                f"{_mean([r['transit'] for r in policy_rows]):>9.0f}"
                f"{_mean([r['coverage'] for r in policy_rows]):>10.0f}"
                f"{_mean([r['resupplies'] for r in policy_rows]):>5.1f}/"
                f"{_mean([r['resupply_steps_distance'] for r in policy_rows]):>4.0f}"
                f"{_mean([r['coverage_ratio'] for r in policy_rows]):>7.1%}"
                f"{sum(1 for r in policy_rows if r['completed']):>4}/{count_instances}"
                f"{_mean(deltas):>14.2f} ± {_ci(deltas):<6.2f}"
            )
            failed = [r["reason"] for r in policy_rows if not r["completed"]]
            if failed:
                print(f"      失败原因: {sorted(set(failed))}")
        print()


if __name__ == "__main__":
    main()
