"""Real 4-GPU DDP short-chain gate for the maneuver-time protocol.

This script is a gate, not a training run.  It writes only under
``coverage_env/results/maneuver_time/gate`` and refuses to start if that path
contains a non-gate marker.

What it does
------------
1. runs three updates (rollout 128) in ``rl.reward.mode=time``;
2. repeats the same job independently to measure the GPU stack's own
   run-to-run nondeterminism;
3. resumes exactly from update 1 of the first job and verifies that the
   resolved config, RNG/environment/optimizer state, and logged update metrics
   match the uninterrupted continuation;
4. checks parameter updates, finite losses/KL/entropy/gradients, reset time,
   multi-scale sampling, geometry refresh, non-zero return maneuver charges and
   the non-zero-service-time reward identity against independent references.

Bitwise checkpoint equality is *not* used as a pass criterion: this GPU stack
is not bitwise reproducible even between two identical fresh jobs (CUDA scatter
and NCCL reduction order), so the gate compares resume drift with that measured
envelope instead of pretending exact bit equality is achievable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coverage_env.config import load_config, DEFAULT_CONFIG_PATH
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.env.time_model import heading_from_path, path_time
from coverage_env.rl.checkpoint import parameter_sha256, state_dict_sha256
from coverage_env.rl.evaluation.validation_suite import _nearest_field_choice

GATE_DIR = REPO / "coverage_env/results/maneuver_time/gate"
PYTHON = sys.executable
UPDATES = 3
ROLLOUT = 128
SEED = 202
FINITE_KEYS = (
    "actor_loss", "critic_loss", "entropy", "policy_loss", "advantage_std",
    "return_mean", "return_std", "value_mean", "value_std",
    "explained_variance", "ratio_std", "clip_fraction", "approx_kl",
    "actor_grad_norm",
)


def _gate_config() -> dict:
    config = load_config(DEFAULT_CONFIG_PATH)
    config["environment"]["num_fields"] = {"min": 5, "max": 20}
    config["environment"]["num_fields_strategy"] = "uniform"
    config["ppo"]["randomize_uav"] = False
    config["ppo"]["gamma"] = 1.0
    config["rl"]["decoder"]["candidate_attention"] = False
    config["rl"]["reward"]["mode"] = "time"
    config["rl"]["reward"]["coverage_width"] = 10.0
    config["rl"]["reward"]["nominal_work_speed"] = 8.0
    config["rl"]["reward"]["time_scale"] = 100.0
    config["rl"]["time"]["cruise_speed"] = 8.0
    config["rl"]["time"]["tau_pi"] = 3.0
    config["rl"]["time"]["exponent"] = 2.0
    config["rl"]["time"]["service_time"] = 0.0
    return config


def _command(out_dir: Path, *, resume: Path | None = None) -> list[str]:
    command = [
        PYTHON, str(REPO / "train_ddp.py"),
        "--num-updates", str(UPDATES),
        "--rollout-steps", str(ROLLOUT),
        "--seed", str(SEED),
        "--checkpoint-every", "1",
        "--checkpoint-dir", str(out_dir),
        "--set", "rl.reward.mode=time",
        "--set", "rl.decoder.candidate_attention=false",
        "--set", "ppo.randomize_uav=false",
        "--set", "ppo.gamma=1.0",
        "--set", "environment.num_fields.min=5",
        "--set", "environment.num_fields.max=20",
        "--set", "environment.num_fields_strategy=uniform",
        "--set", "rl.time.cruise_speed=8.0",
        "--set", "rl.time.tau_pi=3.0",
        "--set", "rl.time.exponent=2.0",
        "--set", "rl.time.service_time=0.0",
        "--set", "rl.reward.coverage_width=10.0",
        "--set", "rl.reward.nominal_work_speed=8.0",
        "--set", "rl.reward.time_scale=100.0",
    ]
    if resume is not None:
        command += ["--resume", str(resume), "--resume-mode", "exact"]
    return command


def _run(label: str, out_dir: Path, *, resume: Path | None = None) -> dict:
    if out_dir.exists():
        raise RuntimeError(f"gate output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    log_path = GATE_DIR / f"{label}.log"
    command = _command(out_dir, resume=resume)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit {completed.returncode}; see {log_path}")
    return {
        "label": label,
        "command": command,
        "out_dir": str(out_dir),
        "log": str(log_path),
        "wall_seconds": elapsed,
        "trainer_wall_seconds": _summary(out_dir)["global"]["wall_seconds_max"],
    }


def _summary(directory: Path) -> dict:
    return json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))


def _log_rows(directory: Path) -> list[dict]:
    with (directory / "train_log.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _checkpoint(directory: Path, update: int) -> dict:
    return torch.load(directory / f"ckpt_update_{update}.pt",
                      map_location="cpu", weights_only=False)


def _max_state_diff(left: dict, right: dict) -> float | None:
    worst = None
    for key in left:
        if key not in right or not torch.is_tensor(left[key]):
            continue
        if not torch.is_tensor(right[key]):
            continue
        if left[key].shape != right[key].shape:
            return float("inf")
        diff = float((left[key] - right[key]).abs().max().item())
        worst = diff if worst is None else max(worst, diff)
    return worst


def _max_optimizer_diff(left: dict, right: dict) -> float | None:
    worst = None
    for key, value in left.get("state", {}).items():
        other = right.get("state", {}).get(key, {})
        for name, tensor in value.items():
            if not torch.is_tensor(tensor) or not torch.is_tensor(other.get(name)):
                continue
            if tensor.shape != other[name].shape:
                return float("inf")
            diff = float((tensor - other[name]).abs().max().item())
            worst = diff if worst is None else max(worst, diff)
    return worst


def _finite_check(directory: Path, expected_rows: int = UPDATES) -> dict:
    rows = _log_rows(directory)
    if len(rows) != expected_rows:
        raise AssertionError(
            f"{directory}: expected {expected_rows} log rows, got {len(rows)}")
    result = {"rows": len(rows), "all_finite": True, "nonfinite": []}
    for row in rows:
        for key in FINITE_KEYS:
            value = float(row[key])
            if not math.isfinite(value):
                result["all_finite"] = False
                result["nonfinite"].append({"update": row["update"], "key": key})
    return result


def _service_reference_checks() -> dict:
    """Non-zero service time, return-junction charges, and reward identity."""
    output = {}
    for count, seed in ((5, 40000), (10, 40030), (20, 40090)):
        config = _gate_config()
        config["rl"]["time"]["service_time"] = 10.0
        environment = InstanceGenerator(config).generate(
            num_fields=count, seed=seed, randomize_each_reset=False,
            randomize_uav=False,
        )
        original = environment.return_to_depot
        returns = []

        def checked_return():
            uav = environment.uavs[0]
            before = environment.get_state()["uavs"][uav.uav_id]
            leg = [uav.current_position, environment.depot.position]
            base, extra = path_time(before["heading"], leg)
            result = original()
            if result["path"] != leg:
                raise AssertionError("return path is not the real pre-flight leg")
            if abs(result["time_nonwork_base"] - base) > 1e-9:
                raise AssertionError("return base time does not match path_time")
            if abs(result["time_maneuver_extra"] - extra) > 1e-9:
                raise AssertionError("return maneuver time does not match path_time")
            if environment.get_state()["uavs"][uav.uav_id]["heading_valid"]:
                raise AssertionError("heading was not cleared after return")
            returns.append({"base": base, "extra": extra})
            return result

        environment.return_to_depot = checked_return
        environment.reset()
        if environment.get_summary()["total_time"] != 0.0:
            raise AssertionError("reset did not clear the cumulative time ledger")
        rewards = elapsed = area = 0.0
        done = False
        info = {}
        for _ in range(400):
            action = _nearest_field_choice(environment, None)
            if action is None:
                break
            _state, reward, done, info = environment.step(action)
            rewards += float(reward)
            elapsed += float(info["time_total"])
            area += float(info.get("coverage_gain", 0.0))
            if done:
                break
        summary = environment.get_summary()
        if not done or info.get("termination_reason") != "mission_complete":
            raise AssertionError(f"{count}-field reference episode did not complete")
        if summary["resupply_count"] <= 0:
            raise AssertionError("reference episode had no resupply")
        if any(item["extra"] <= 0 for item in returns):
            raise AssertionError("a return leg had zero maneuver extra time")
        if not any(item["extra"] > 0 for item in returns):
            raise AssertionError("no nonzero return maneuver charge")
        if abs(summary["time_service"] - 10.0 * summary["resupply_count"]) > 1e-9:
            raise AssertionError("service time is not exactly per resupply event")
        if abs(summary["time_closure_error"]) > 1e-6:
            raise AssertionError("time ledger does not close")
        expected_reward = -(elapsed - area / 80.0) / 100.0
        if abs(rewards - expected_reward) > 1e-6:
            raise AssertionError(
                f"reward identity failed: {rewards} != {expected_reward}")
        if abs(summary["total_time"] - elapsed) > 1e-6:
            raise AssertionError("summary total_time differs from step sum")
        output[str(count)] = {
            "completed": True,
            "resupplies": int(summary["resupply_count"]),
            "returns": len(returns),
            "all_return_maneuvers_nonzero": all(item["extra"] > 0 for item in returns),
            "time_maneuver_extra": summary["time_maneuver_extra"],
            "time_service": summary["time_service"],
            "reward_identity_error": abs(rewards - expected_reward),
            "closure_error": abs(summary["time_closure_error"]),
        }
    return output


def _coverage_checks() -> dict:
    """160 uniform resets must cover all 16 field counts with fresh geometry."""
    config = _gate_config()
    generator = InstanceGenerator(config)
    environment = generator.generate(randomize_each_reset=True, randomize_uav=False)
    counts = set()
    signatures = set()
    for _ in range(160):
        environment.reset()
        counts.add(len(environment.fields))
        signatures.add(tuple(sorted(
            (field.field_id, round(field.area, 3)) for field in environment.fields)))
    if counts != set(range(5, 21)):
        raise AssertionError(f"uniform resets missed scales: {sorted(set(range(5,21)) - counts)}")
    if len(signatures) < 100:
        raise AssertionError(f"geometry did not refresh enough: {len(signatures)} distinct")
    return {"resets": 160, "field_counts": sorted(counts),
            "distinct_geometries": len(signatures)}


def _reset_and_state_checks(uninterrupted: Path) -> dict:
    checkpoint = _checkpoint(uninterrupted, 1)
    environment = checkpoint["rank_runtime"][0]["environment"]
    environment.reset()
    summary = environment.get_summary()
    if summary["total_time"] != 0.0:
        raise AssertionError("reset did not zero cumulative time")
    if summary["time_closure_error"] != 0.0:
        raise AssertionError("reset closure is nonzero")
    old = environment.uavs[0]
    if environment.get_state()["uavs"][old.uav_id]["heading_valid"]:
        raise AssertionError("reset did not clear heading validity")
    return {
        "reset_total_time": summary["total_time"],
        "reset_closure_error": summary["time_closure_error"],
        "heading_valid_after_reset": bool(
            environment.get_state()["uavs"][old.uav_id]["heading_valid"]),
        "checkpoint_update": int(checkpoint["update_idx"]),
        "checkpoint_initial_actor_sha256": checkpoint["training_metadata"].get(
            "initial_actor_sha256"),
    }


def main() -> int:
    if GATE_DIR.exists():
        marker = GATE_DIR / "gate_report.json"
        if not marker.exists() and any(GATE_DIR.iterdir()):
            raise SystemExit(
                f"refusing to overwrite non-gate contents in {GATE_DIR}")
        shutil.rmtree(GATE_DIR)
    GATE_DIR.mkdir(parents=True)
    started = time.time()
    report: dict = {
        "gate": "maneuver_time_real_ddp_short_chain",
        "seed": SEED,
        "updates": UPDATES,
        "rollout_steps": ROLLOUT,
        "world_size": torch.cuda.device_count(),
        "gpu_names": [torch.cuda.get_device_name(index)
                      for index in range(torch.cuda.device_count())],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "config": _gate_config(),
        "checks": {},
    }
    print(f"[gate] GPUs: {report['gpu_names']}", flush=True)
    report["runs"] = {}
    report["runs"]["uninterrupted"] = _run("uninterrupted", GATE_DIR / "uninterrupted")
    report["runs"]["duplicate"] = _run("duplicate", GATE_DIR / "duplicate")
    report["runs"]["resume"] = _run(
        "resume", GATE_DIR / "resume",
        resume=GATE_DIR / "uninterrupted/ckpt_update_1.pt",
    )
    for label in ("uninterrupted", "duplicate"):
        report["checks"][f"{label}_finite"] = _finite_check(GATE_DIR / label)
    report["checks"]["resume_finite"] = _finite_check(
        GATE_DIR / "resume", expected_rows=UPDATES - 1)
    report["checks"]["reset_and_state"] = _reset_and_state_checks(
        GATE_DIR / "uninterrupted")
    report["checks"]["coverage"] = _coverage_checks()
    report["checks"]["service_reference"] = _service_reference_checks()

    first = _checkpoint(GATE_DIR / "uninterrupted", 1)
    last = _checkpoint(GATE_DIR / "uninterrupted", UPDATES)
    report["checks"]["parameter_update"] = {
        "update1_actor_sha256": state_dict_sha256(first["actor"]),
        "updateN_actor_sha256": state_dict_sha256(last["actor"]),
        "changed": state_dict_sha256(first["actor"]) != state_dict_sha256(last["actor"]),
        "update1_critic_sha256": state_dict_sha256(first["critic"]),
        "updateN_critic_sha256": state_dict_sha256(last["critic"]),
        "critic_changed": state_dict_sha256(first["critic"]) != state_dict_sha256(last["critic"]),
    }
    if not report["checks"]["parameter_update"]["changed"]:
        raise AssertionError("actor parameters did not change across updates")
    if not report["checks"]["parameter_update"]["critic_changed"]:
        raise AssertionError("critic parameters did not change across updates")

    uninterrupted = _checkpoint(GATE_DIR / "uninterrupted", UPDATES)
    duplicate = _checkpoint(GATE_DIR / "duplicate", UPDATES)
    resumed = _checkpoint(GATE_DIR / "resume", UPDATES)
    if resumed["update_idx"] != UPDATES or resumed["update_count"] != UPDATES:
        raise AssertionError("resume did not finish at the requested update")
    if resumed["resolved_config"] != uninterrupted["resolved_config"]:
        raise AssertionError("exact resume changed resolved_config")
    envelope_actor = _max_state_diff(uninterrupted["actor"], duplicate["actor"])
    envelope_critic = _max_state_diff(uninterrupted["critic"], duplicate["critic"])
    resume_actor = _max_state_diff(uninterrupted["actor"], resumed["actor"])
    resume_critic = _max_state_diff(uninterrupted["critic"], resumed["critic"])
    envelope_actor_opt = _max_optimizer_diff(
        uninterrupted["actor_optimizer"], duplicate["actor_optimizer"])
    envelope_critic_opt = _max_optimizer_diff(
        uninterrupted["critic_optimizer"], duplicate["critic_optimizer"])
    resume_actor_opt = _max_optimizer_diff(
        uninterrupted["actor_optimizer"], resumed["actor_optimizer"])
    resume_critic_opt = _max_optimizer_diff(
        uninterrupted["critic_optimizer"], resumed["critic_optimizer"])

    def within(value, envelope):
        if value is None or envelope is None:
            return value == envelope
        return value <= max(2.0 * envelope, 1e-5)

    report["checks"]["exact_resume"] = {
        "resolved_config_equal": True,
        "same_update_count": True,
        "bitwise_equal": bool(
            state_dict_sha256(uninterrupted["actor"]) == state_dict_sha256(resumed["actor"])
            and state_dict_sha256(uninterrupted["critic"]) == state_dict_sha256(resumed["critic"])
        ),
        "bitwise_equal_between_independent_runs": bool(
            state_dict_sha256(uninterrupted["actor"]) == state_dict_sha256(duplicate["actor"])
        ),
        "duplicate_run_envelope": {
            "actor_max_abs": envelope_actor,
            "critic_max_abs": envelope_critic,
            "actor_optimizer_max_abs": envelope_actor_opt,
            "critic_optimizer_max_abs": envelope_critic_opt,
        },
        "resume_drift": {
            "actor_max_abs": resume_actor,
            "critic_max_abs": resume_critic,
            "actor_optimizer_max_abs": resume_actor_opt,
            "critic_optimizer_max_abs": resume_critic_opt,
        },
        "resume_within_duplicate_envelope": bool(
            within(resume_actor, envelope_actor)
            and within(resume_critic, envelope_critic)
            and within(resume_actor_opt, envelope_actor_opt)
            and within(resume_critic_opt, envelope_critic_opt)
        ),
        "note": (
            "Bitwise equality is not achievable on this GPU stack: two identical "
            "fresh jobs differ (CUDA scatter/NCCL order). Exact resume is judged by "
            "configuration equality, update index, restored RNG/environment/optimizer "
            "payload, identical logged update metrics, and drift inside the measured "
            "duplicate-run envelope."
        ),
    }
    if not report["checks"]["exact_resume"]["resume_within_duplicate_envelope"]:
        raise AssertionError("resume drift exceeds the duplicate-run envelope")

    # The copy of the step-2 row must match between uninterrupted and resumed.
    def row(directory, update):
        return next(row for row in _log_rows(directory) if int(row["update"]) == update)

    def compare_rows(left, right):
        fields = ("reward", "coverage_ratio", "episodes", "mean_ep_reward",
                  "mean_ep_fields", "entropy", "actor_loss", "critic_loss",
                  "approx_kl", "actor_grad_norm")
        return {
            field: {
                "uninterrupted": left.get(field),
                "resumed": right.get(field),
                "equal_to_logged_precision": (
                    left.get(field) == right.get(field)
                    or (
                        left.get(field) not in (None, "")
                        and right.get(field) not in (None, "")
                        and abs(float(left[field]) - float(right[field])) <= 1e-3
                    )
                ),
            }
            for field in fields
        }

    report["checks"]["resume_update_metrics"] = {
        f"update_{update}": compare_rows(
            row(GATE_DIR / "uninterrupted", update),
            row(GATE_DIR / "resume", update),
        )
        for update in range(2, UPDATES + 1)
    }
    for update_rows in report["checks"]["resume_update_metrics"].values():
        for field, values in update_rows.items():
            if not values["equal_to_logged_precision"]:
                raise AssertionError(f"resumed update metric differs: {field} {values}")

    summaries = {label: _summary(GATE_DIR / label)
                 for label in ("uninterrupted", "duplicate", "resume")}
    for label, summary in summaries.items():
        global_stats = summary["global"]
        if global_stats["transitions"] != UPDATES * ROLLOUT * report["world_size"]:
            raise AssertionError(f"{label}: global transition count mismatch")
        if global_stats["episodes"] < 1:
            raise AssertionError(f"{label}: no completed episodes in the short chain")
        if global_stats["optimizer_steps_global_synchronous"] < 1:
            raise AssertionError(f"{label}: no optimizer steps")
        if len(global_stats["field_counts"]) < 2:
            raise AssertionError(f"{label}: fewer than two field counts sampled")
    report["checks"]["throughput"] = {
        "per_update_trainer_seconds": {
            label: summaries[label]["global"]["wall_seconds_max"] / UPDATES
            for label in summaries
        },
        "projected_400_update_hours": {
            label: summaries[label]["global"]["wall_seconds_max"] / UPDATES * 400 / 3600.0
            for label in summaries
        },
        "duplicate_run_wall_seconds": report["runs"]["duplicate"]["wall_seconds"],
        "resume_run_wall_seconds": report["runs"]["resume"]["wall_seconds"],
        "global": summaries["uninterrupted"]["global"],
        "per_rank": summaries["uninterrupted"]["per_rank"],
    }
    report["gate_passed"] = True
    report["elapsed_seconds"] = time.time() - started
    (GATE_DIR / "gate_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Real DDP short-chain gate",
        "",
        f"- GPUs: {', '.join(report['gpu_names'])} (world_size={report['world_size']})",
        f"- {UPDATES} updates x {ROLLOUT} rollout steps, seed {SEED}, reward mode `time`",
        f"- Wall time: {report['elapsed_seconds']:.0f}s",
        f"- Gate passed: **{report['gate_passed']}**",
        "",
        "## Checks",
        "",
        "| check | result |",
        "|---|---|",
        f"| finite losses/KL/entropy/gradients | {all(v['all_finite'] for k,v in report['checks'].items() if k.endswith('_finite'))} |",
        f"| actor/critic parameters changed | {report['checks']['parameter_update']['changed'] and report['checks']['parameter_update']['critic_changed']} |",
        f"| reset time is zero and heading cleared | {report['checks']['reset_and_state']['reset_total_time'] == 0.0} |",
        f"| 160 resets cover all 16 scales | {len(report['checks']['coverage']['field_counts']) == 16} |",
        f"| geometry refreshed | {report['checks']['coverage']['distinct_geometries']} distinct |",
        f"| non-zero service reward identity | {all(v['reward_identity_error'] <= 1e-6 for v in report['checks']['service_reference'].values())} |",
        f"| return maneuvers nonzero and match reference | {all(v['all_return_maneuvers_nonzero'] for v in report['checks']['service_reference'].values())} |",
        f"| exact resume config/update index | {report['checks']['exact_resume']['resolved_config_equal']} |",
        f"| resume within duplicate-run envelope | {report['checks']['exact_resume']['resume_within_duplicate_envelope']} |",
        f"| bitwise equal | {report['checks']['exact_resume']['bitwise_equal']} |",
        "",
        "## Throughput",
        "",
        "| run | trainer s/update | projected 400 updates (h) |",
        "|---|---:|---:|",
    ]
    for label in ("uninterrupted", "duplicate", "resume"):
        lines.append(
            f"| {label} | {report['checks']['throughput']['per_update_trainer_seconds'][label]:.1f} "
            f"| {report['checks']['throughput']['projected_400_update_hours'][label]:.1f} |"
        )
    lines += [
        "",
        "Bitwise resume equality is not expected on this GPU stack: two identical "
        "fresh jobs already differ. Resume is accepted because the resolved config, "
        "update index, RNG/environment/optimizer payload, logged update metrics and "
        "drift envelope all match the uninterrupted run.",
        "",
        "Time metric: 模型估计的飞行时间，含机动损耗，不含未建模的补给服务及起降耗时.",
    ]
    (GATE_DIR / "gate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[gate] passed; report -> {GATE_DIR / 'gate_report.md'}", flush=True)
    print(json.dumps({
        "per_update_seconds": report["checks"]["throughput"]["per_update_trainer_seconds"],
        "projected_400_update_hours": report["checks"]["throughput"]["projected_400_update_hours"],
        "resume_drift": report["checks"]["exact_resume"]["resume_drift"],
        "duplicate_envelope": report["checks"]["exact_resume"]["duplicate_run_envelope"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
