"""Start the six formal maneuver-time A/B training runs.

Order (serial, same four GPUs, same DDP world size for every job)::

    A/seed42, B/seed42, A/seed43, B/seed43, A/seed44, B/seed44

``A`` uses ``rl.reward.mode=distance`` (the historical normalised distance
objective); ``B`` uses ``rl.reward.mode=time`` (the fixed-scale maneuver-time
objective).  Everything else is identical: same network, same frozen UAV
capability, same field distribution, same optimizer budget, same seed.  The
driver records the source SHA, resolved config, commands, hardware, data split,
initial parameter hashes and the gate report before starting the first run.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from coverage_env.config import DEFAULT_CONFIG_PATH, load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.checkpoint import parameter_sha256, state_dict_sha256
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.decoder.global_state_encoder import build_uav_feature
from coverage_env.rl.trainer.distributed import resolve_config

FORMAL_DIR = REPO / "coverage_env/results/maneuver_time/formal_fixed"
DEFAULT_GATE_REPORT = REPO / "coverage_env/results/maneuver_time/value_gate/value_gate_report.json"
PYTHON = sys.executable
SEEDS = (42, 43, 44)
UPDATES = 400
ROLLOUT = 128
CHECKPOINT_EVERY = 20
RUN_ORDER = [(mode, seed) for seed in SEEDS for mode in ("A", "B")]


def _common_overrides(mode: str) -> list[str]:
    reward_mode = "distance" if mode == "A" else "time"
    return [
        f"rl.reward.mode={reward_mode}",
        "rl.decoder.candidate_attention=false",
        "ppo.randomize_uav=false",
        "ppo.gamma=1.0",
        "environment.num_fields.min=5",
        "environment.num_fields.max=20",
        "environment.num_fields_strategy=uniform",
        "uav.speed=8.0",
        "uav.coverage_width=10.0",
        "rl.reward.coverage_width=10.0",
        "rl.reward.nominal_work_speed=8.0",
        "rl.reward.time_scale=100.0",
        "rl.time.cruise_speed=8.0",
        "rl.time.tau_pi=3.0",
        "rl.time.exponent=2.0",
        "rl.time.service_time=0.0",
    ]


def _run_dir(mode: str, seed: int) -> Path:
    return FORMAL_DIR / f"{mode}_seed{seed}"


def _resolve(mode: str, seed: int) -> dict:
    args = SimpleNamespace(
        num_fields=None, set=_common_overrides(mode), ppo_epochs=None,
        batch_size=None, gae_lambda=None,
    )
    return resolve_config(args)


def _assert_effective(config: dict, mode: str) -> None:
    """Assert values the environment/trainer will actually receive."""
    expected_reward = "distance" if mode == "A" else "time"
    checks = {
        "environment.num_fields.min": config["environment"]["num_fields"]["min"],
        "environment.num_fields.max": config["environment"]["num_fields"]["max"],
        "environment.num_fields_strategy": config["environment"]["num_fields_strategy"],
        "ppo.randomize_uav": config["ppo"]["randomize_uav"],
        "ppo.gamma": config["ppo"]["gamma"],
        "uav.speed": config["uav"]["speed"],
        "uav.coverage_width": config["uav"]["coverage_width"],
        "rl.reward.mode": config["rl"]["reward"]["mode"],
        "rl.reward.coverage_width": config["rl"]["reward"]["coverage_width"],
        "rl.reward.nominal_work_speed": config["rl"]["reward"]["nominal_work_speed"],
        "rl.reward.time_scale": config["rl"]["reward"]["time_scale"],
        "rl.time.cruise_speed": config["rl"]["time"]["cruise_speed"],
        "rl.time.tau_pi": config["rl"]["time"]["tau_pi"],
        "rl.time.exponent": config["rl"]["time"]["exponent"],
        "rl.time.service_time": config["rl"]["time"]["service_time"],
        "rl.decoder.candidate_attention": config["rl"]["decoder"]["candidate_attention"],
    }
    expected = {
        "environment.num_fields.min": 5,
        "environment.num_fields.max": 20,
        "environment.num_fields_strategy": "uniform",
        "ppo.randomize_uav": False,
        "ppo.gamma": 1.0,
        "uav.speed": 8.0,
        "uav.coverage_width": 10.0,
        "rl.reward.mode": expected_reward,
        "rl.reward.coverage_width": 10.0,
        "rl.reward.nominal_work_speed": 8.0,
        "rl.reward.time_scale": 100.0,
        "rl.time.cruise_speed": 8.0,
        "rl.time.tau_pi": 3.0,
        "rl.time.exponent": 2.0,
        "rl.time.service_time": 0.0,
        "rl.decoder.candidate_attention": False,
    }
    failures = []
    for key, expected_value in expected.items():
        actual = checks[key]
        if isinstance(expected_value, float):
            ok = abs(float(actual) - expected_value) <= 1e-12
        else:
            ok = actual == expected_value
        if not ok:
            failures.append(f"{key}={actual!r} (expected {expected_value!r})")
    if failures:
        raise SystemExit("effective config mismatch: " + "; ".join(failures))


def _initial_hash(mode: str, seed: int) -> str:
    """Reproduce ``_worker``'s initialisation: manual_seed then CoveragePolicy."""
    config = _resolve(mode, seed)
    torch.manual_seed(int(seed))
    policy = CoveragePolicy(None, config=config)
    return parameter_sha256(policy)


def _preflight(gpu_count: int) -> dict:
    if gpu_count != 4:
        raise SystemExit(
            f"this protocol is frozen for world_size=4, found {gpu_count} visible GPUs"
        )
    preflight = {"world_size": gpu_count, "runs": {}, "configs": {}}
    feature_dim = int(build_uav_feature({
        "position": (0.0, 0.0), "max_energy": 1500.0, "max_payload": 50.0,
        "speed": 8.0, "coverage_width": 10.0, "remaining_energy": 1500.0,
        "remaining_payload": 50.0, "alpha": 0.08, "beta": 0.02,
        "payload_per_area": 0.003,
    }).shape[0])
    if feature_dim != 16:
        raise SystemExit(
            f"UAV global feature dim is {feature_dim}, expected the 16-d heading-aware input"
        )
    preflight["uav_feature_dim"] = feature_dim
    for mode, seed in RUN_ORDER:
        config = _resolve(mode, seed)
        _assert_effective(config, mode)
        policy = CoveragePolicy(None, config=config)
        if int(policy.global_state_encoder.uav_dim) != feature_dim:
            raise SystemExit("GlobalStateEncoder.uav_dim is not the 16-d UAV input")
        preflight["configs"][f"{mode}_seed{seed}"] = config
        preflight["runs"][f"{mode}_seed{seed}"] = {
            "reward_mode": config["rl"]["reward"]["mode"],
            "initial_actor_sha256": _initial_hash(mode, seed),
            "num_updates": UPDATES,
            "rollout_steps": ROLLOUT,
            "checkpoint_every": CHECKPOINT_EVERY,
            "world_size": gpu_count,
        }
    for seed in SEEDS:
        left = preflight["runs"][f"A_seed{seed}"]["initial_actor_sha256"]
        right = preflight["runs"][f"B_seed{seed}"]["initial_actor_sha256"]
        if left != right:
            raise SystemExit(
                f"paired seed {seed}: A/B initial parameter hashes differ")
        preflight["runs"][f"A_seed{seed}"]["paired_initial_hash"] = left
        preflight["runs"][f"B_seed{seed}"]["paired_initial_hash"] = left
    return preflight


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True).strip()


def _gpu_memory_mib() -> list[int]:
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def _wait_for_idle_gpus(limit_mib: int = 2000, timeout: float = 120.0) -> None:
    started = time.time()
    while True:
        used = _gpu_memory_mib()
        if len(used) != 4:
            raise SystemExit(f"expected 4 GPUs, nvidia-smi reported {len(used)}")
        if max(used) <= limit_mib:
            return
        if time.time() - started > timeout:
            raise SystemExit(f"GPUs still busy before launch: {used} MiB")
        time.sleep(5.0)


def _command(mode: str, seed: int, out_dir: Path,
             resume: Path | None = None) -> list[str]:
    command = [
        PYTHON, str(REPO / "train_ddp.py"),
        "--num-updates", str(UPDATES),
        "--rollout-steps", str(ROLLOUT),
        "--seed", str(seed),
        "--checkpoint-every", str(CHECKPOINT_EVERY),
        "--checkpoint-dir", str(out_dir),
    ]
    for item in _common_overrides(mode):
        command += ["--set", item]
    if resume is not None:
        command += ["--resume", str(resume), "--resume-mode", "exact"]
    return command


def _latest_checkpoint(directory: Path) -> Path | None:
    checkpoints = []
    for path in directory.glob("ckpt_update_*.pt"):
        try:
            checkpoints.append((int(path.stem.rsplit("_", 1)[-1]), path))
        except ValueError:
            continue
    return max(checkpoints)[1] if checkpoints else None


def _validate_run(mode: str, seed: int, directory: Path,
                  preflight: dict, resumed_from: Path | None = None) -> dict:
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    metadata = summary["training_metadata"]
    expected_hash = preflight["runs"][f"{mode}_seed{seed}"]["initial_actor_sha256"]
    if resumed_from is None:
        if metadata.get("initial_actor_sha256") != expected_hash:
            raise SystemExit(
                f"{mode}_seed{seed}: run initial hash "
                f"{metadata.get('initial_actor_sha256')} != preflight {expected_hash}")
    else:
        saved = torch.load(resumed_from, map_location="cpu", weights_only=False)
        resume_hash = state_dict_sha256(saved["actor"])
        if metadata.get("initial_actor_sha256") != resume_hash:
            raise SystemExit(
                f"{mode}_seed{seed}: resumed initial hash "
                f"{metadata.get('initial_actor_sha256')} != checkpoint {resume_hash}")
    if metadata.get("num_updates") != UPDATES:
        raise SystemExit(f"{mode}_seed{seed}: run summary target is not {UPDATES}")
    global_stats = summary["global"]
    expected_transitions = UPDATES * ROLLOUT * preflight["world_size"]
    if global_stats["transitions"] != expected_transitions:
        raise SystemExit(
            f"{mode}_seed{seed}: {global_stats['transitions']} transitions, "
            f"expected {expected_transitions}")
    checkpoints = sorted(directory.glob("ckpt_update_*.pt"))
    expected_checkpoints = UPDATES // CHECKPOINT_EVERY
    if len(checkpoints) != expected_checkpoints:
        raise SystemExit(
            f"{mode}_seed{seed}: {len(checkpoints)} checkpoints, "
            f"expected {expected_checkpoints}")
    resolved_path = directory / "resolved_config_update_1.json"
    if resolved_path.exists():
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        saved = resolved["resolved_config"]
        if saved != preflight["configs"][f"{mode}_seed{seed}"]:
            raise SystemExit(f"{mode}_seed{seed}: saved resolved_config differs from preflight")
    return {
        "run_summary": str(directory / "run_summary.json"),
        "final_checkpoint": str(directory / f"ckpt_update_{UPDATES}.pt"),
        "checkpoints": len(checkpoints),
        "global": global_stats,
        "initial_actor_sha256": metadata["initial_actor_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    global FORMAL_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-sha", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", default=None,
                        choices=[f"{mode}_seed{seed}" for mode, seed in RUN_ORDER])
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--out-root", default=str(FORMAL_DIR))
    parser.add_argument("--gate-report", default=str(DEFAULT_GATE_REPORT))
    parser.add_argument("--task-timeout", type=float, default=8 * 3600)
    args = parser.parse_args(argv)
    FORMAL_DIR = Path(args.out_root)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_count = len([item for item in args.gpu_ids.split(",") if item.strip()])
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if dirty:
        raise SystemExit("working tree is dirty; freeze and commit the code before launching")
    if args.expected_sha and not sha.startswith(args.expected_sha):
        raise SystemExit(f"HEAD {sha} does not match expected {args.expected_sha}")
    gate_path = Path(args.gate_report)
    if not gate_path.exists():
        raise SystemExit(f"missing fixed-code value gate report: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not (gate.get("gate_passed") or gate.get("passed")):
        raise SystemExit("the fixed-code value gate did not pass; refusing to start formal training")
    preflight = _preflight(gpu_count)
    preflight["git_sha"] = sha
    preflight["git_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    preflight["gate_report"] = str(gate_path)
    preflight["value_gate"] = gate
    preflight["hardware"] = {
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "visible_devices": args.gpu_ids,
    }
    preflight["data_split"] = {
        "validation_seeds": "40000-40095 (6 per scale 5-20; reused for development/selection)",
        "test_seeds": "60000-61599 (100 per scale 5-20; new sealed block for the corrected rerun)",
        "historical_used_test": "50000-51599 (already used by the invalidated 869bdd4 run; not for the rerun)",
        "training_instance_sources": "continuous uniform resampling of 5-20 fields; no validation/test seed is reused",
    }
    preflight["budget"] = {
        "updates": UPDATES,
        "rollout_steps": ROLLOUT,
        "checkpoint_every": CHECKPOINT_EVERY,
        "world_size": gpu_count,
        "per_run_global_transitions": UPDATES * ROLLOUT * gpu_count,
        "per_run_global_optimizer_steps": UPDATES * (ROLLOUT // 64) * 4,
    }
    preflight["selection_rule"] = (
        "completion count first; if and only if every validation instance is complete, "
        "compare scale-equal mean total_time on all 96 fixed instances; "
        "otherwise report the reliability gate failed and select the earliest update "
        "among the highest completion count. One unified checkpoint per training seed."
    )
    preflight["runs_to_start"] = [
        f"{mode}_seed{seed}" for mode, seed in RUN_ORDER
    ]
    FORMAL_DIR.mkdir(parents=True, exist_ok=True)
    (FORMAL_DIR / "preflight_and_manifest.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "sha": sha,
        "world_size": gpu_count,
        "initial_hashes": {
            key: value["initial_actor_sha256"]
            for key, value in preflight["runs"].items()
        },
        "gate_passed": True,
        "out_root": str(FORMAL_DIR),
    }, indent=1))
    if args.dry_run:
        return 0

    selected = set(args.only) if args.only else None
    status = {
        "sha": sha, "started": time.time(), "runs": {},
        "preflight": str(FORMAL_DIR / "preflight_and_manifest.json"),
    }
    status_path = FORMAL_DIR / "formal_status.json"
    for mode, seed in RUN_ORDER:
        label = f"{mode}_seed{seed}"
        if selected is not None and label not in selected:
            continue
        out_dir = _run_dir(mode, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / f"ckpt_update_{UPDATES}.pt"
        if final.exists() and (out_dir / "run_summary.json").exists():
            status["runs"][label] = _validate_run(mode, seed, out_dir, preflight)
            status["runs"][label]["skipped"] = "already complete"
            status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            continue
        _wait_for_idle_gpus()
        resume = _latest_checkpoint(out_dir)
        command = _command(mode, seed, out_dir, resume=resume)
        log_path = out_dir / "train.log"
        started = time.time()
        print(f"[formal] starting {label} -> {out_dir} (resume={resume})", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": str(REPO)},
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.task_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    returncode = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = 124
        elapsed = time.time() - started
        record = {
            "command": command,
            "log": str(log_path),
            "started": started,
            "elapsed_seconds": elapsed,
            "exit_code": returncode,
            "resumed_from": str(resume) if resume else None,
        }
        if returncode != 0:
            record["status"] = "failed"
            status["runs"][label] = record
            status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            raise SystemExit(
                f"{label} failed with exit {returncode}; see {log_path}")
        record.update(_validate_run(mode, seed, out_dir, preflight,
                                    resumed_from=resume))
        record["status"] = "complete"
        status["runs"][label] = record
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"[formal] completed {label} in {elapsed / 3600.0:.2f}h", flush=True)
    status["finished"] = time.time()
    status["all_complete"] = all(
        status["runs"].get(f"{mode}_seed{seed}", {}).get("status") == "complete"
        for mode, seed in RUN_ORDER
    )
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps({"all_complete": status["all_complete"],
                      "status": str(status_path)}, indent=1))
    return 0 if status["all_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
