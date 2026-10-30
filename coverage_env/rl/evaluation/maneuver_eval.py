"""Fixed-set validation selection and final test for the maneuver-time protocol.

The protocol fixes two disjoint seed blocks:

* validation ``40000-40095`` -- 96 instances, six per field count 5..20;
* final test ``50000-51599`` -- 1600 instances, 100 per field count.

Every number in this module is produced from those fixed seeds.  The final
block is never read by ``validate`` and must only be evaluated once, after the
checkpoint has been selected on the validation block.

Checkpoint selection follows the frozen rule, in this order:

1. more completed validation episodes wins;
2. only if every candidate completes all 96 instances may the scale-equal mean
   ``total_time`` on the full fixed set be compared;
3. if no candidate completes all 96, the reliability gate is reported as failed
   and the earliest update among the highest completion count is selected;
4. candidates tied on completion but not complete are never ranked by mean time
   over their different successful subsets -- the earlier update breaks the tie.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.checkpoint import evaluation_config
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice

FIELD_COUNTS = tuple(range(5, 21))
VALIDATION_BASE = 40000
VALIDATION_PER_COUNT = 6
TEST_BASE = 50000
TEST_PER_COUNT = 100
CHECKPOINT_EVERY = 20
TOTAL_UPDATES = 400
MAX_STEPS = 400
TIME_METRIC_NAME = (
    "模型估计的飞行时间，含机动损耗，不含未建模的补给服务及起降耗时"
)

_POLICY_CACHE: dict[str, tuple[CoveragePolicy, dict, torch.device]] = {}
_CONFIG: dict | None = None
_DEVICE = torch.device("cpu")
_GPU_IDS: list[int] = []


# ---------------------------------------------------------------------------
# Fixed seed blocks
# ---------------------------------------------------------------------------
def protocol_seeds(base: int, per_count: int) -> dict[int, list[int]]:
    """Return ``{field_count: [seeds]}`` using the frozen contiguous block."""
    seeds: dict[int, list[int]] = {}
    offset = 0
    for count in FIELD_COUNTS:
        seeds[count] = list(range(base + offset, base + offset + per_count))
        offset += per_count
    return seeds


def validation_seeds() -> dict[int, list[int]]:
    return protocol_seeds(VALIDATION_BASE, VALIDATION_PER_COUNT)


def test_seeds() -> dict[int, list[int]]:
    return protocol_seeds(TEST_BASE, TEST_PER_COUNT)


def expected_checkpoint_updates() -> list[int]:
    return list(range(CHECKPOINT_EVERY, TOTAL_UPDATES + 1, CHECKPOINT_EVERY))


def protocol_config() -> dict:
    """Execution config shared by every evaluated policy.

    Reward mode is deliberately not part of the execution environment: it only
    changes the scalar returned by ``calculate_reward``.  The protocol fixes the
    UAV capability, field range and network flags; those are asserted here so a
    future default change cannot silently alter the frozen evaluation.
    """
    config = load_config(DEFAULT_CONFIG_PATH)
    config["environment"]["num_fields"] = {"min": 5, "max": 20}
    config["environment"]["num_fields_strategy"] = "uniform"
    config["ppo"]["randomize_uav"] = False
    config["ppo"]["gamma"] = 1.0
    config["rl"]["reward"]["mode"] = "distance"
    config["rl"]["reward"]["coverage_width"] = 10.0
    config["rl"]["reward"]["nominal_work_speed"] = 8.0
    config["rl"]["reward"]["time_scale"] = 100.0
    config["rl"]["time"]["cruise_speed"] = 8.0
    config["rl"]["time"]["tau_pi"] = 3.0
    config["rl"]["time"]["exponent"] = 2.0
    config["rl"]["time"]["service_time"] = 0.0
    config["rl"]["decoder"]["candidate_attention"] = False
    config["uav"]["speed"] = 8.0
    config["uav"]["coverage_width"] = 10.0
    return config


def _check_protocol_config(config: dict) -> None:
    """Fail loudly if a frozen execution value is not what the protocol says."""
    failures = []
    for path, expected in [
        (("environment", "num_fields", "min"), 5),
        (("environment", "num_fields", "max"), 20),
        (("environment", "num_fields_strategy"), "uniform"),
        (("ppo", "randomize_uav"), False),
        (("ppo", "gamma"), 1.0),
        (("uav", "speed"), 8.0),
        (("uav", "coverage_width"), 10.0),
        (("rl", "reward", "coverage_width"), 10.0),
        (("rl", "reward", "nominal_work_speed"), 8.0),
        (("rl", "reward", "time_scale"), 100.0),
        (("rl", "time", "cruise_speed"), 8.0),
        (("rl", "time", "tau_pi"), 3.0),
        (("rl", "time", "exponent"), 2.0),
        (("rl", "time", "service_time"), 0.0),
        (("rl", "decoder", "candidate_attention"), False),
    ]:
        node = config
        for part in path:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        value = node if not isinstance(node, dict) else None
        if isinstance(expected, float):
            ok = value is not None and math.isclose(float(value), expected, abs_tol=1e-12)
        else:
            ok = value == expected
        if not ok:
            failures.append(f"{'.'.join(path)}={value!r}, expected {expected!r}")
    if failures:
        raise ValueError("Frozen protocol config mismatch: " + "; ".join(failures))


# ---------------------------------------------------------------------------
# Policy loading and episode execution
# ---------------------------------------------------------------------------
def _device_for_worker() -> torch.device:
    if _GPU_IDS and torch.cuda.is_available():
        return torch.device(f"cuda:{_GPU_IDS[_worker_index() % len(_GPU_IDS)]}")
    return torch.device("cpu")


def _worker_index() -> int:
    identity = getattr(mp.current_process(), "_identity", None)
    return max(0, int(identity[0]) - 1) if identity else 0


def _init_worker(gpu_ids: list[int]) -> None:
    global _GPU_IDS
    _GPU_IDS = list(gpu_ids)
    torch.set_num_threads(1)
    if _GPU_IDS and torch.cuda.is_available():
        torch.cuda.set_device(_GPU_IDS[_worker_index() % len(_GPU_IDS)])


def load_policy(path: str | Path) -> tuple[CoveragePolicy, dict, torch.device]:
    """Load one checkpoint as a greedy evaluation policy, caching per process."""
    key = str(Path(path).resolve())
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    saved = torch.load(key, map_location="cpu", weights_only=False)
    config = evaluation_config(saved, protocol_config())
    _check_protocol_config(config)
    policy = CoveragePolicy(None, config=config)
    policy.load_state_dict(saved["actor"])
    device = _device_for_worker()
    policy.to(device)
    policy.eval()
    value = (policy, config, device)
    _POLICY_CACHE[key] = value
    return value


def _nn_choice(environment, _state):
    started = time.perf_counter()
    action = _nearest_field_choice(environment, None)
    return action, time.perf_counter() - started, 0.0


def _policy_choice_factory(policy: CoveragePolicy):
    def choose(environment, state):
        # Time the cheap mask first, then the graph/decoder forward with the
        # already-enumerated candidates so mask cost is not hidden inside
        # "inference".  The action is still the greedy argmax of the joint
        # distribution -- the same deployment rule used by the training gate.
        mask_started = time.perf_counter()
        actions = environment.get_action_candidates()
        mask = environment.get_action_mask(actions)
        mask_seconds = time.perf_counter() - mask_started
        infer_started = time.perf_counter()
        with torch.no_grad():
            output = policy(
                state, environment=environment,
                candidate_actions=actions, feasible_mask=mask,
            )
            action = CoveragePolicy.greedy_action(output)
        infer_seconds = time.perf_counter() - infer_started
        return action, mask_seconds, infer_seconds
    return choose


def run_episode(environment, choose) -> dict:
    """Execute one fixed instance and return episode-level metrics."""
    state = environment.reset()
    done = False
    info: dict = {}
    steps = 0
    failed_steps = 0
    failure_reasons: list[str] = []
    energy_violations = 0
    payload_violations = 0
    mask_seconds = 0.0
    infer_seconds = 0.0
    while steps < MAX_STEPS:
        action, mask_s, infer_s = choose(environment, state)
        mask_seconds += float(mask_s)
        infer_seconds += float(infer_s)
        if action is None:
            break
        state, _reward, done, info = environment.step(action)
        steps += 1
        if info.get("failed"):
            failed_steps += 1
            failure_reasons.append(str(info.get("failure_reason", "")))
        for uav in environment.uavs:
            if float(uav.remaining_energy) < -1e-6:
                energy_violations += 1
            if float(uav.remaining_payload) < -1e-6:
                payload_violations += 1
        if done:
            break
    summary = environment.get_summary()
    completed = bool(done and info.get("success")
                     and info.get("termination_reason") == "mission_complete")
    total_time = float(summary.get("total_time", 0.0))
    maneuver = float(summary.get("time_maneuver_extra", 0.0))
    return {
        "completed": completed,
        "termination_reason": str(info.get("termination_reason", "no_action")),
        "steps": steps,
        "failed_steps": failed_steps,
        "failure_reasons": failure_reasons[-5:],
        "energy_violations": energy_violations,
        "payload_violations": payload_violations,
        "total_time": total_time,
        "time_work_base": float(summary.get("time_work_base", 0.0)),
        "time_nonwork_base": float(summary.get("time_nonwork_base", 0.0)),
        "time_maneuver_extra": maneuver,
        "time_service": float(summary.get("time_service", 0.0)),
        "time_resupply_or_return": float(summary.get("time_resupply_or_return", 0.0)),
        "time_closure_error": float(summary.get("time_closure_error", 0.0)),
        "maneuver_share": maneuver / total_time if total_time > 0 else 0.0,
        "total_distance": float(summary.get("total_distance", 0.0)),
        "coverage_ratio": float(summary.get("coverage_ratio", 0.0)),
        "total_energy": float(summary.get("total_energy", 0.0)),
        "resupplies": int(summary.get("resupply_count", 0)),
        "visits": int(summary.get("execution_count", 0)),
        "returned_to_depot": bool(summary.get("returned_to_depot", False)),
        "remaining_energy": float(summary.get("remaining_energy", {}).get("UAV-1", 0.0))
        if isinstance(summary.get("remaining_energy"), dict) else 0.0,
        "steps_timed": steps,
        "mask_seconds": mask_seconds,
        "infer_seconds": infer_seconds,
        "mask_ms_per_step": 1000.0 * mask_seconds / steps if steps else 0.0,
        "infer_ms_per_step": 1000.0 * infer_seconds / steps if steps else 0.0,
    }


def _make_environment(config: dict, count: int, seed: int):
    return InstanceGenerator(config).generate(
        num_fields=count, seed=seed, randomize_each_reset=False,
        randomize_uav=False,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def discover_checkpoints(run_dir: str | Path) -> dict[int, Path]:
    paths = sorted(Path(run_dir).glob("ckpt_update_*.pt"))
    found: dict[int, Path] = {}
    for path in paths:
        try:
            update = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        found[update] = path
    expected = expected_checkpoint_updates()
    missing = [update for update in expected if update not in found]
    extra = [update for update in found if update not in expected]
    if missing or extra:
        raise SystemExit(
            f"checkpoint set in {run_dir} does not match the protocol: "
            f"missing={missing}, unexpected={extra}"
        )
    return {update: found[update] for update in expected}


def _validation_task(task):
    checkpoint, count, seed = task
    policy, config, _device = load_policy(checkpoint)
    environment = _make_environment(config, count, seed)
    result = run_episode(environment, _policy_choice_factory(policy))
    result.update(field_count=count, seed=seed, checkpoint=str(checkpoint))
    return result


def candidate_statistics(rows: list[dict], total_instances: int) -> list[dict]:
    """Aggregate validation rows per checkpoint update."""
    by_update: dict[int, list[dict]] = {}
    for row in rows:
        update = int(row["checkpoint_update"])
        by_update.setdefault(update, []).append(row)
    candidates = []
    for update in sorted(by_update):
        subset = by_update[update]
        completed_rows = [row for row in subset if row["completed"]]
        scale_means = {}
        for count in FIELD_COUNTS:
            values = [row["total_time"] for row in completed_rows
                      if row["field_count"] == count]
            if values:
                scale_means[count] = float(statistics.mean(values))
        all_complete = (
            len(completed_rows) == total_instances
            and len(scale_means) == len(FIELD_COUNTS)
        )
        candidates.append({
            "update": update,
            "completed": len(completed_rows),
            "total": total_instances,
            "completion_rate": len(completed_rows) / total_instances,
            "all_complete": all_complete,
            "scale_mean_total_time": scale_means if all_complete else {},
            "mean_total_time": (
                float(statistics.mean(scale_means.values())) if all_complete else None
            ),
            "per_scale_completed": {
                str(count): sum(1 for row in subset
                                if row["field_count"] == count and row["completed"])
                for count in FIELD_COUNTS
            },
        })
    return candidates


def select_candidate(candidates: list[dict], total_instances: int) -> dict:
    """Apply the frozen completion-first selection rule.

    ``candidates`` must contain one entry per checkpoint update.  The return
    value includes the ``reliability_gate_passed`` flag; when it is false no
    mean-time comparison is made between candidates.
    """
    if not candidates:
        raise ValueError("no candidates to select from")
    complete = [c for c in candidates if c["all_complete"]]
    if complete:
        # Full fixed set, scale-equal mean, then earlier update.
        chosen = min(complete, key=lambda c: (c["mean_total_time"], c["update"]))
        return {
            "selected_update": chosen["update"],
            "selected_completed": chosen["completed"],
            "selected_total": total_instances,
            "reliability_gate_passed": True,
            "rule": "all_complete_then_scale_equal_mean_total_time",
            "candidates": candidates,
        }
    best_completed = max(c["completed"] for c in candidates)
    tied = [c for c in candidates if c["completed"] == best_completed]
    chosen = min(tied, key=lambda c: c["update"])
    return {
        "selected_update": chosen["update"],
        "selected_completed": chosen["completed"],
        "selected_total": total_instances,
        "reliability_gate_passed": False,
        "rule": "completion_count_then_earlier_update_no_time_comparison",
        "candidates": candidates,
    }


def _write_validation_markdown(selection: dict, path: Path) -> None:
    lines = [
        "# Validation checkpoint selection",
        "",
        f"- Validation seeds: `{VALIDATION_BASE}-{VALIDATION_BASE + 15 * VALIDATION_PER_COUNT + VALIDATION_PER_COUNT - 1}`"
        f" ({len(FIELD_COUNTS) * VALIDATION_PER_COUNT} instances, "
        f"{VALIDATION_PER_COUNT} per field count 5-20)",
        f"- Selected update: **{selection['selected_update']}**",
        f"- Completed: **{selection['selected_completed']}/{selection['selected_total']}**",
        f"- Reliability gate passed: **{selection['reliability_gate_passed']}**",
        f"- Rule: `{selection['rule']}`",
        "",
        "| update | completed | all complete | scale-equal mean total_time (s) |",
        "|---:|---:|:---:|---:|",
    ]
    for candidate in selection["candidates"]:
        mean = candidate["mean_total_time"]
        lines.append(
            f"| {candidate['update']} | {candidate['completed']}/{candidate['total']} "
            f"| {'yes' if candidate['all_complete'] else 'no'} "
            f"| {'-' if mean is None else f'{mean:.3f}'} |"
        )
    lines += [
        "",
        "No mean time is compared unless every candidate row marked `all complete`; "
        "when none is complete the earliest update among the highest completion count "
        "is selected and the reliability gate is reported as failed.",
        "",
        "Time metric: " + TIME_METRIC_NAME + ".",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_validate(args) -> int:
    _check_protocol_config(protocol_config())
    checkpoints = discover_checkpoints(args.run_dir)
    seeds = validation_seeds()
    tasks = [
        (str(checkpoints[update]), count, seed)
        for update in expected_checkpoint_updates()
        for count in FIELD_COUNTS
        for seed in seeds[count]
    ]
    out_dir = Path(args.out_dir or args.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict] = []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(args.gpu_ids,)) as pool:
        for index, row in enumerate(pool.imap_unordered(_validation_task, tasks, chunksize=1), 1):
            update = int(Path(row["checkpoint"]).stem.rsplit("_", 1)[-1])
            row["checkpoint_update"] = update
            row.pop("checkpoint", None)
            rows.append(row)
            if index % 120 == 0 or index == len(tasks):
                print(f"[validate] {index}/{len(tasks)} episodes "
                      f"({time.time() - started:.0f}s)", flush=True)
    rows.sort(key=lambda row: (row["checkpoint_update"], row["field_count"], row["seed"]))
    candidates = candidate_statistics(rows, len(tasks) // len(expected_checkpoint_updates()))
    selection = select_candidate(candidates, len(FIELD_COUNTS) * VALIDATION_PER_COUNT)
    selection["selected_checkpoint"] = str(checkpoints[selection["selected_update"]].resolve())
    selection["run_dir"] = str(Path(args.run_dir).resolve())
    selection["validation_seed_block"] = [VALIDATION_BASE,
                                          VALIDATION_BASE + 96 - 1]
    selection["metric"] = TIME_METRIC_NAME
    selection["elapsed_seconds"] = time.time() - started
    (out_dir / "validation_rows.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "selection.json").write_text(
        json.dumps(selection, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_validation_markdown(selection, out_dir / "selection.md")
    print(json.dumps({
        "selected_update": selection["selected_update"],
        "completed": f"{selection['selected_completed']}/{selection['selected_total']}",
        "reliability_gate_passed": selection["reliability_gate_passed"],
    }, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# Final test
# ---------------------------------------------------------------------------
def _final_task(task):
    count, seed, specs = task
    template = _make_environment(_CONFIG, count, seed)
    results = {}
    for label, checkpoint in specs:
        environment = deepcopy(template)
        if checkpoint is None:
            results[label] = run_episode(environment, _nn_choice)
        else:
            policy, _config, _device = load_policy(checkpoint)
            results[label] = run_episode(environment, _policy_choice_factory(policy))
    return {"field_count": count, "seed": seed, "results": results}


def _mean(values):
    return float(statistics.mean(values)) if values else None


def _median(values):
    return float(statistics.median(values)) if values else None


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return float(ordered[index])


def _bootstrap_ci(values, seed: int = 0, samples: int = 10000):
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if len(array) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def _stratified_bootstrap(per_scale_values: dict, seed: int = 0,
                          samples: int = 10000):
    """Bootstrap instances within each scale, then average scale means equally."""
    arrays = {count: np.asarray([v for v in values if v is not None], dtype=float)
              for count, values in per_scale_values.items()}
    arrays = {count: array for count, array in arrays.items() if len(array) > 0}
    if not arrays:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        means = []
        for array in arrays.values():
            means.append(float(array[rng.integers(0, len(array), size=len(array))].mean()))
        draws.append(float(np.mean(means)))
    low, high = np.percentile(draws, [2.5, 97.5])
    return [float(low), float(high)]


def _metric_block(rows: list[dict]) -> dict:
    completed = [row for row in rows if row["completed"]]
    return {
        "n": len(rows),
        "n_completed": len(completed),
        "completion_rate": len(completed) / len(rows) if rows else 0.0,
        "total_time_mean": _mean([row["total_time"] for row in completed]),
        "total_time_median": _median([row["total_time"] for row in completed]),
        "total_time_p95": _p95([row["total_time"] for row in completed]),
        "time_work_base_mean": _mean([row["time_work_base"] for row in completed]),
        "time_nonwork_base_mean": _mean([row["time_nonwork_base"] for row in completed]),
        "time_maneuver_extra_mean": _mean([row["time_maneuver_extra"] for row in completed]),
        "time_service_mean": _mean([row["time_service"] for row in completed]),
        "time_resupply_or_return_mean": _mean([row["time_resupply_or_return"] for row in completed]),
        "maneuver_share_mean": _mean([row["maneuver_share"] for row in completed]),
        "total_distance_mean": _mean([row["total_distance"] for row in rows]),
        "coverage_ratio_mean": _mean([row["coverage_ratio"] for row in rows]),
        "total_energy_mean": _mean([row["total_energy"] for row in rows]),
        "resupplies_mean": _mean([row["resupplies"] for row in rows]),
        "visits_mean": _mean([row["visits"] for row in rows]),
        "failed_steps_total": int(sum(row["failed_steps"] for row in rows)),
        "energy_violations_total": int(sum(row["energy_violations"] for row in rows)),
        "payload_violations_total": int(sum(row["payload_violations"] for row in rows)),
        "return_infeasible_count": int(sum(
            1 for row in rows if row["termination_reason"] == "return_infeasible")),
        "mask_ms_per_step_mean": _mean([row["mask_ms_per_step"] for row in rows]),
        "infer_ms_per_step_mean": _mean([
            row["infer_ms_per_step"] for row in rows if row["infer_ms_per_step"] > 0]),
        "time_closure_error_max_abs": max(
            (abs(row["time_closure_error"]) for row in rows), default=0.0),
    }


def _paired_values(rows: list[dict], baseline: str, candidate: str):
    values = []
    for row in rows:
        left = row["results"].get(baseline)
        right = row["results"].get(candidate)
        if left and right and left["completed"] and right["completed"]:
            values.append(left["total_time"] - right["total_time"])
    return values


def _paired_block(rows: list[dict], baseline: str, candidate: str) -> dict:
    values = _paired_values(rows, baseline, candidate)
    completion_left = sum(1 for row in rows if row["results"][baseline]["completed"])
    completion_right = sum(1 for row in rows if row["results"][candidate]["completed"])
    return {
        "baseline": baseline,
        "candidate": candidate,
        "n_pairs": len(values),
        "paired_mean": _mean(values),
        "paired_median": _median(values),
        "ci95": _bootstrap_ci(values),
        "wins": sum(1 for value in values if value > 1e-6),
        "ties": sum(1 for value in values if abs(value) <= 1e-6),
        "losses": sum(1 for value in values if value < -1e-6),
        "baseline_completed": completion_left,
        "candidate_completed": completion_right,
        "completion_delta": completion_right - completion_left,
    }


def _summarize_final(rows: list[dict], labels: list[str], seeds: list[int]) -> dict:
    summary: dict = {"labels": labels, "scales": {}, "overall": {}, "seeds": seeds}
    for count in FIELD_COUNTS:
        subset = [row for row in rows if row["field_count"] == count]
        by_label = {
            label: _metric_block([row["results"][label] for row in subset])
            for label in labels
        }
        nn = "NN"
        nn_pairs = {
            label: _paired_block(subset, nn, label)
            for label in labels if label != nn
        }
        ab_pairs = {}
        for seed in seeds:
            a_label = f"A_seed{seed}"
            b_label = f"B_seed{seed}"
            if a_label in labels and b_label in labels:
                ab_pairs[f"{a_label}_vs_{b_label}"] = _paired_block(
                    subset, a_label, b_label)
        summary["scales"][str(count)] = {
            "n": len(subset),
            "labels": by_label,
            "nn_minus_policy": nn_pairs,
            "a_minus_b": ab_pairs,
        }
    # Overall scale-equal metrics and stratified paired CIs.
    overall_labels = {}
    for label in labels:
        per_scale = {count: [row["results"][label] for row in rows
                             if row["field_count"] == count]
                     for count in FIELD_COUNTS}
        blocks = {count: _metric_block(per_scale[count]) for count in FIELD_COUNTS}
        overall_labels[label] = {
            key: _mean([blocks[count][key] for count in FIELD_COUNTS
                        if blocks[count][key] is not None])
            for key in ("total_time_mean", "completion_rate", "total_distance_mean",
                        "coverage_ratio_mean", "total_energy_mean",
                        "resupplies_mean", "visits_mean", "maneuver_share_mean",
                        "time_maneuver_extra_mean", "time_service_mean",
                        "mask_ms_per_step_mean", "infer_ms_per_step_mean")
        }
        overall_labels[label]["n"] = sum(blocks[count]["n"] for count in FIELD_COUNTS)
        overall_labels[label]["n_completed"] = sum(
            blocks[count]["n_completed"] for count in FIELD_COUNTS)
    overall_pairs = {}
    for label in labels:
        if label == "NN":
            continue
        per_scale_values = {
            count: _paired_values([row for row in rows if row["field_count"] == count],
                                  "NN", label)
            for count in FIELD_COUNTS
        }
        overall_pairs[f"NN_vs_{label}"] = {
            "n_pairs": sum(len(values) for values in per_scale_values.values()),
            "paired_mean_scale_equal": _mean([
                _mean(values) for values in per_scale_values.values()
                if values
            ]),
            "ci95": _stratified_bootstrap(per_scale_values),
        }
    for seed in seeds:
        a_label = f"A_seed{seed}"
        b_label = f"B_seed{seed}"
        if a_label not in labels or b_label not in labels:
            continue
        per_scale_values = {
            count: _paired_values([row for row in rows if row["field_count"] == count],
                                  a_label, b_label)
            for count in FIELD_COUNTS
        }
        overall_pairs[f"{a_label}_vs_{b_label}"] = {
            "n_pairs": sum(len(values) for values in per_scale_values.values()),
            "paired_mean_scale_equal": _mean([
                _mean(values) for values in per_scale_values.values() if values
            ]),
            "ci95": _stratified_bootstrap(per_scale_values),
        }
    summary["overall"] = {"labels": overall_labels, "pairs": overall_pairs}
    return summary


def _write_final_markdown(summary: dict, path: Path) -> None:
    labels = summary["labels"]
    lines = [
        "# Final test report",
        "",
        f"- Test seeds: `{TEST_BASE}-{TEST_BASE + 15 * TEST_PER_COUNT + TEST_PER_COUNT - 1}`"
        f" ({len(FIELD_COUNTS) * TEST_PER_COUNT} instances, {TEST_PER_COUNT} per field count 5-20).",
        "- Time metric: **" + TIME_METRIC_NAME + "**.",
        "- Sign convention: **baseline - candidate**; a positive value means the "
        "candidate is better (shorter model-estimated flight time).",
        "- Paired time differences use only instances where both policies completed; "
        "completion counts are reported separately. Confidence intervals resample "
        "instances (not time steps), and overall CIs resample within each scale and "
        "average scales equally.",
        "",
        "## Per-scale model time and completion",
        "",
        "| scale | policy | completed | total_time mean (s) | work | nonwork | maneuver | service | distance (m) | coverage | energy | resupplies | visits |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in FIELD_COUNTS:
        block = summary["scales"][str(count)]["labels"]
        for label in labels:
            m = block[label]
            def fmt(value, digits=1):
                return "-" if value is None else f"{value:.{digits}f}"
            lines.append(
                f"| {count} | {label} | {m['n_completed']}/{m['n']} | "
                f"{fmt(m['total_time_mean'])} | {fmt(m['time_work_base_mean'])} | "
                f"{fmt(m['time_nonwork_base_mean'])} | {fmt(m['time_maneuver_extra_mean'])} | "
                f"{fmt(m['time_service_mean'])} | {fmt(m['total_distance_mean'])} | "
                f"{fmt(m['coverage_ratio_mean'], 4)} | {fmt(m['total_energy_mean'])} | "
                f"{fmt(m['resupplies_mean'], 2)} | {fmt(m['visits_mean'], 2)} |"
            )
    lines += ["", "## Per-scale paired differences (baseline - candidate, seconds)",
              "",
              "| scale | comparison | n pairs | mean | median | 95% CI | win/tie/loss | completion delta |",
              "|---:|---|---:|---:|---:|---|---:|---:|"]
    for count in FIELD_COUNTS:
        section = summary["scales"][str(count)]
        for comparison in list(section["nn_minus_policy"].values()) + list(section["a_minus_b"].values()):
            ci = comparison["ci95"]
            ci_text = "-" if ci[0] is None else f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"
            mean = "-" if comparison["paired_mean"] is None else f"{comparison['paired_mean']:+.1f}"
            median = "-" if comparison["paired_median"] is None else f"{comparison['paired_median']:+.1f}"
            lines.append(
                f"| {count} | {comparison['baseline']} - {comparison['candidate']} | "
                f"{comparison['n_pairs']} | {mean} | {median} | {ci_text} | "
                f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} | "
                f"{comparison['completion_delta']:+d} |"
            )
    lines += ["", "## Overall (scale-equal)", "",
              "| policy | completion | total_time mean (s) | maneuver share | distance (m) | coverage | energy | resupplies | visits | mask ms/step | infer ms/step |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        m = summary["overall"]["labels"][label]
        def fmt(value, digits=1):
            return "-" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {label} | {fmt(m['completion_rate'] * 100, 1)}% | "
            f"{fmt(m['total_time_mean'])} | {fmt(m['maneuver_share_mean'] * 100, 1)}% | "
            f"{fmt(m['total_distance_mean'])} | {fmt(m['coverage_ratio_mean'], 4)} | "
            f"{fmt(m['total_energy_mean'])} | {fmt(m['resupplies_mean'], 2)} | "
            f"{fmt(m['visits_mean'], 2)} | {fmt(m['mask_ms_per_step_mean'], 2)} | "
            f"{fmt(m['infer_ms_per_step_mean'], 2)} |"
        )
    lines += ["", "## Overall paired differences (baseline - candidate, seconds)", "",
              "| comparison | n pairs | scale-equal mean | 95% stratified CI |",
              "|---|---:|---:|---|"]
    for name, comparison in summary["overall"]["pairs"].items():
        mean = comparison["paired_mean_scale_equal"]
        ci = comparison["ci95"]
        ci_text = "-" if ci[0] is None else f"[{ci[0]:+.1f}, {ci[1]:+.1f}]"
        lines.append(
            f"| {name} | {comparison['n_pairs']} | "
            f"{'-' if mean is None else f'{mean:+.1f}'} | {ci_text} |"
        )
    lines += ["", "Inference and mask timings are measured in the evaluation workers "
              "(GPU inference when available; mask/candidate enumeration on CPU). "
              "They are diagnostic, not part of the protocol budget.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def command_final(args) -> int:
    global _CONFIG
    _CONFIG = protocol_config()
    _check_protocol_config(_CONFIG)
    specs = []
    seeds = set()
    for item in args.selection:
        if "=" not in item:
            raise SystemExit(f"--selection expects LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        checkpoint = payload["selected_checkpoint"]
        if not Path(checkpoint).exists():
            raise SystemExit(f"selected checkpoint does not exist: {checkpoint}")
        specs.append((label, checkpoint))
        for part in label.split("_"):
            if part.startswith("seed") and part[4:].isdigit():
                seeds.add(int(part[4:]))
    labels = ["NN"] + [label for label, _checkpoint in specs]
    if len(labels) != len(set(labels)):
        raise SystemExit("duplicate labels in --selection")
    policy_specs = [("NN", None)] + list(specs)
    seeds = sorted(seeds)
    tasks = [(count, seed, policy_specs)
             for count in FIELD_COUNTS for seed in test_seeds()[count]]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=args.workers, initializer=_init_worker,
                  initargs=(args.gpu_ids,)) as pool:
        for index, row in enumerate(pool.imap_unordered(_final_task, tasks, chunksize=1), 1):
            rows.append(row)
            if index % 40 == 0 or index == len(tasks):
                print(f"[final] {index}/{len(tasks)} instances "
                      f"({time.time() - started:.0f}s)", flush=True)
    rows.sort(key=lambda row: (row["field_count"], row["seed"]))
    summary = _summarize_final(rows, labels, seeds)
    summary["test_seed_block"] = [TEST_BASE, TEST_BASE + 1600 - 1]
    summary["metric"] = TIME_METRIC_NAME
    summary["elapsed_seconds"] = time.time() - started
    summary["checkpoints"] = {label: checkpoint for label, checkpoint in specs}
    (out_dir / "final_rows.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "final_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_final_markdown(summary, out_dir / "final_report.md")
    print(f"[final] wrote {out_dir / 'final_report.md'}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="evaluate every checkpoint on the fixed validation set")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--out-dir", default=None)
    validate.add_argument("--workers", type=int, default=12)
    validate.add_argument("--gpu-ids", type=int, nargs="*", default=[])
    validate.set_defaults(func=command_validate)

    final = sub.add_parser("final", help="evaluate selected policies on the sealed test set")
    final.add_argument("--selection", action="append", required=True,
                       metavar="LABEL=selection.json")
    final.add_argument("--out-dir", required=True)
    final.add_argument("--workers", type=int, default=16)
    final.add_argument("--gpu-ids", type=int, nargs="*", default=[])
    final.set_defaults(func=command_final)

    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
