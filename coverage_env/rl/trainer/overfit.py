"""Fixed-instance PPO fine-tuning, reusing the audited rollout and hard gates.

prepare: freeze ten 20-field instances from the initial checkpoint's config.
train: repeatedly train on that bank; evaluate ALL training instances at update
0 and every checkpoint. These are in-sample diagnostics, never sealed tests.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import datetime
import shutil
import json
import math
import os
from pathlib import Path
import random
import statistics

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from coverage_env.config import load_config
from coverage_env.env.fixed_instance_pool import (
    FixedInstancePool, file_sha256, load_bank, write_bank,
)
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.checkpoint import capture_rng_state, restore_rng_state, parameter_sha256
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.evaluation.maneuver_eval import (
    run_episode, _nn_choice, _policy_choice_factory,
)
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.distributed import DistributedTrainer, _free_port, validate_exact_resume


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def prepare(args):
    if args.checkpoint:
        path = Path(args.checkpoint).resolve()
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if not saved.get("resolved_config"):
            raise ValueError("Initial checkpoint must contain resolved_config; no guessed defaults.")
        config = deepcopy(saved["resolved_config"])
        initialization = {"checkpoint": str(path), "sha256": file_sha256(path)}
    else:
        config = load_config()
        initialization = None
    if args.count < 1 or args.fields < 1:
        raise ValueError("count and fields must be positive.")
    if float(config["ppo"].get("gamma", 0)) != 1.0:
        raise ValueError("This diagnostic requires the existing gamma=1 protocol.")
    if config["ppo"].get("advantage_mode", "gae") != "gae":
        raise ValueError("Use a GAE PPO checkpoint for this diagnostic.")
    config["environment"]["num_fields"] = {"min": args.fields, "max": args.fields}
    config["ppo"]["randomize_uav"] = False
    config["ppo"]["value_gate"] = True
    seeds = list(range(args.instance_seed, args.instance_seed + args.count))
    # An overfit set is training data: never repurpose previous validation or
    # final-test blocks. Default 70000..70009 is declared before any results.
    if any(40000 <= seed <= 61599 for seed in seeds):
        raise ValueError("Do not use the prior validation/final-test seed blocks for this bank.")
    environments = []
    for seed in seeds:
        environments.append(InstanceGenerator(config).generate(
            num_fields=args.fields, seed=seed, randomize_each_reset=False, randomize_uav=False))
        print(f"froze instance {seed}: {args.fields} fields", flush=True)
    manifest = write_bank(args.bank, config, environments, seeds, initialization=initialization)
    print(f"Bank frozen: {len(manifest['instances'])} instances; no training started.")


def aggregate_rows(rows, references=None):
    """No success-only averages; incomplete runs cannot look artificially cheap."""
    if not rows or len({r["index"] for r in rows}) != len(rows):
        raise ValueError("Expected exactly one evaluation row per fixed instance.")
    complete = sum(bool(r["completed"]) for r in rows)
    result = {"instances": len(rows), "completed": complete,
              "failed_steps": sum(r["failed_steps"] for r in rows)}
    for key in ("total_time", "total_distance", "total_energy", "coverage_ratio",
                "visits", "resupplies", "time_work_base", "time_nonwork_base",
                "time_maneuver_extra", "time_service"):
        result[key] = statistics.mean(r[key] for r in rows) if complete == len(rows) else None
    for name, baseline in (references or {}).items():
        by_index = {r["index"]: r for r in baseline}
        if set(by_index) != {r["index"] for r in rows}:
            raise ValueError("Reference and candidate instance identities do not match.")
        comparable = complete == len(rows) and all(r["completed"] for r in baseline)
        for metric in ("total_time", "total_distance"):
            diffs = [by_index[r["index"]][metric] - r[metric] for r in rows] if comparable else []
            result[f"{name}_minus_candidate_{metric}"] = statistics.mean(diffs) if diffs else None
        if comparable:
            differences = [by_index[r["index"]]["total_time"] - r["total_time"] for r in rows]
            result[f"{name}_time_improvement_percent"] = statistics.mean(
                100 * (by_index[r["index"]]["total_time"] - r["total_time"])
                / by_index[r["index"]]["total_time"] for r in rows)
            result[f"{name}_wins"] = sum(d > 1e-6 for d in differences)
            result[f"{name}_ties"] = sum(abs(d) <= 1e-6 for d in differences)
            result[f"{name}_losses"] = sum(d < -1e-6 for d in differences)
    return result


def candidate_key(summary):
    """Complete-set time, completion-first; never rank partial success subsets."""
    all_done = summary["completed"] == summary["instances"]
    return (-summary["completed"], summary["total_time"] if all_done else math.inf,
            summary["update"])


class OverfitTrainer(DistributedTrainer):
    """Add fixed-bank evaluation/checkpoint metadata without replacing PPO."""
    def _save_checkpoint(self, update_idx):
        super()._save_checkpoint(update_idx)
        self.evaluate_bank(update_idx)

    def evaluate_bank(self, update_idx):
        # Evaluation must not advance training action RNG, the pool's schedule,
        # or mutable mission state. Each rank handles disjoint pristine copies.
        rng = capture_rng_state()
        actor = self.agent.actor.module
        was_training = actor.training
        local = {"rows": [], "nn": [], "error": None,
                 "usage": self.environment.usage()}
        try:
            actor.eval()
            manifest, templates = load_bank(self.environment.directory,
                                           expected_manifest_sha=self.environment.manifest_sha)
            choose = _policy_choice_factory(actor)
            for index in range(self.rank, len(templates), self.world_size):
                env = deepcopy(templates[index])
                row = run_episode(env, choose)
                row.update(index=index, seed=manifest["instances"][index]["seed"])
                self._check_eval_row(row)
                local["rows"].append(row)
                if update_idx == 0:
                    baseline = run_episode(deepcopy(templates[index]), _nn_choice)
                    baseline.update(index=index, seed=row["seed"])
                    self._check_eval_row(baseline)
                    if not baseline["completed"]:
                        raise RuntimeError(f"NN cannot complete fixed instance {index}; do not silently replace it.")
                    local["nn"].append(baseline)
        except Exception as exc:
            local["error"] = repr(exc)
        finally:
            actor.train(was_training)
            restore_rng_state(rng)
        gathered = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if any(item["error"] for item in gathered):
            raise RuntimeError(f"Fixed-bank evaluation failed: {[g['error'] for g in gathered]}")
        error = None
        if self.rank == 0:
            try:
                self._write_evaluation(update_idx, gathered)
            except Exception as exc:
                error = repr(exc)
        errors = [error]
        if dist.is_initialized():
            dist.broadcast_object_list(errors, src=0)
        if errors[0]:
            raise RuntimeError(f"Cannot save evaluation: {errors[0]}")

    @staticmethod
    def _check_eval_row(row):
        if any(not math.isfinite(float(row[k])) for k in
               ("total_time", "total_distance", "time_closure_error")):
            raise RuntimeError("Nonfinite fixed-bank evaluation metric.")
        if row["energy_violations"] or row["payload_violations"] or abs(row["time_closure_error"]) > 1e-6:
            raise RuntimeError("Resource or time-ledger violation in fixed-bank evaluation.")

    def _write_evaluation(self, update_idx, gathered):
        directory = Path(self.checkpoint_dir)
        rows = sorted([r for g in gathered for r in g["rows"]], key=lambda r: r["index"])
        expected = list(range(len(self.environment.manifest["instances"])))
        if [r["index"] for r in rows] != expected:
            raise RuntimeError("Evaluation did not cover the entire fixed bank exactly once.")
        if update_idx == 0:
            references = {"initial": rows, "nn": sorted(
                [r for g in gathered for r in g["nn"]], key=lambda r: r["index"])}
            _write_json(directory / "references.json", references)
        else:
            references = json.loads((directory / "references.json").read_text(encoding="utf-8"))
        summary = aggregate_rows(rows, references)
        summary.update(update=update_idx, scope="training_set_only",
                       checkpoint=f"ckpt_update_{update_idx}.pt")
        _write_json(directory / "evaluation" / f"update_{update_idx}.json",
                    {"summary": summary, "rows": rows,
                     "rank_usage": [g["usage"] for g in gathered]})
        best_path = directory / "best.json"
        best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
        if best is None or candidate_key(summary) < candidate_key(best):
            _write_json(best_path, summary)
        # Rebuild from immutable per-checkpoint rows; a resumed evaluation does
        # not append duplicate points or silently omit update 0.
        history = [json.loads(p.read_text(encoding="utf-8"))["summary"]
                   for p in (directory / "evaluation").glob("update_*.json")]
        history.sort(key=lambda r: r["update"])
        keys = sorted(set().union(*(r.keys() for r in history)))
        with (directory / "learning_curve.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)
        print(f"[fixed-bank] update={update_idx} complete={summary['completed']}/{len(rows)} "
              f"time={summary['total_time']} NN gain={summary.get('nn_minus_candidate_total_time')} "
              f"initial gain={summary.get('initial_minus_candidate_total_time')}", flush=True)


class _LocalModule(torch.nn.Module):
    """Expose the DDP .module contract for bounded single-process CPU tests."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _worker(rank, world_size, args):
    distributed = args.device == "cuda"
    if distributed:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size,
                                timeout=datetime.timedelta(minutes=15))
    try:
        torch.set_num_threads(1)
        device = torch.device(f"cuda:{rank}" if distributed else "cpu")
        environment = FixedInstancePool(args.bank, seed=args.seed + rank)
        manifest = environment.manifest
        config = deepcopy(manifest["config"])
        torch.manual_seed(args.seed)
        agent = PPOAgent(CoveragePolicy(environment, config=config), config, device=device)
        start = 1
        runtime = None
        saved = None
        branch_mode = bool(getattr(args, "branch_from", None))
        initialization = manifest.get("initialization")
        if branch_mode:
            saved = torch.load(args.branch_from, map_location="cpu", weights_only=False)
            metadata = saved.get("training_metadata", {})
            if metadata.get("bank_manifest_sha") != environment.manifest_sha:
                raise ValueError("Branch checkpoint must use the same fixed instance bank.")
            for key, expected in (("num_updates", args.num_updates),
                                  ("rollout_steps", args.rollout_steps),
                                  ("seed", args.seed),
                                  ("world_size", world_size)):
                if metadata.get(key) != expected:
                    raise ValueError(f"Branch checkpoint {key}={metadata.get(key)} != {expected}")
            config = deepcopy(saved["resolved_config"])
            config["ppo"]["actor_learning_rate"] = float(args.actor_lr)
            if float(config["ppo"].get("learning_rate", args.actor_lr)) == float(args.actor_lr):
                pass
            start = agent.load_checkpoint(args.branch_from, restore_rng=False,
                                          load_optimizers=True) + 1
            runtime = saved["rank_runtime"][rank]
            environment = runtime["environment"]
            environment.directory = str(Path(args.bank).resolve())
            # Preserve critic LR and all optimizer moments; change only actor LR.
            agent.actor_lr_start = float(args.actor_lr)
            for group in agent.actor_optimizer.param_groups:
                group["lr"] = float(args.actor_lr)
            if rank == 0:
                source_references = Path(args.branch_from).resolve().parent / "references.json"
                if not source_references.exists():
                    raise ValueError("Branch source has no references.json")
                shutil.copy(source_references, Path(args.output) / "references.json")
        elif args.resume:
            saved = torch.load(args.resume, map_location="cpu", weights_only=False)
            validate_exact_resume(saved, config, args, world_size)
            if saved["training_metadata"].get("bank_manifest_sha") != environment.manifest_sha:
                raise ValueError("Exact resume must use the same fixed instance bank.")
            start = agent.load_checkpoint(args.resume, restore_rng=False) + 1
            runtime = saved["rank_runtime"][rank]
            environment = runtime["environment"]
            environment.directory = str(Path(args.bank).resolve())
        elif initialization:
            path = initialization["checkpoint"]
            if file_sha256(path) != initialization["sha256"]:
                raise ValueError("Initial checkpoint changed after bank preparation.")
            agent.load_checkpoint(path, restore_rng=False, load_optimizers=False)
            agent.update_count = 0
        initial_hash = parameter_sha256(agent.actor)
        agent.actor = DDP(agent.actor, device_ids=[rank]) if distributed else _LocalModule(agent.actor)
        agent.critic = DDP(agent.critic, device_ids=[rank]) if distributed else agent.critic
        trainer = OverfitTrainer(environment, agent, rank=rank, world_size=world_size,
                                 checkpoint_dir=args.output, checkpoint_every=args.eval_every,
                                 config=config)
        trainer.run_metadata = {"seed": args.seed, "num_updates": args.num_updates,
                                "rollout_steps": args.rollout_steps, "world_size": world_size,
                                "bank_manifest_sha": environment.manifest_sha,
                                "initial_actor_sha256": initial_hash,
                                "initialization": initialization,
                                "purpose": "training_set_overfit_not_generalization"}
        if branch_mode:
            trainer.run_metadata.update({
                "branch_from": str(Path(args.branch_from).resolve()),
                "branch_actor_lr": float(getattr(args, "actor_lr")),
                "branch_label": getattr(args, "branch_label", None),
                "branch_source_update": int(saved.get("update_count", 0)),
                "branch_source_actor_sha256": parameter_sha256(agent.actor),
                "purpose": "paired_branch_actor_lr_experiment_not_exact_resume",
            })
        if runtime:
            if branch_mode:
                trainer._rank_stats = deepcopy(runtime["rank_stats"])
                trainer._count_emas = deepcopy(runtime["count_emas"])
            else:
                trainer.run_metadata = deepcopy(saved["training_metadata"])
                trainer._rank_stats = deepcopy(runtime["rank_stats"])
                trainer._count_emas = deepcopy(runtime["count_emas"])
            restore_rng_state(runtime["rng"])
        else:
            random.seed(args.seed + rank)
            np.random.seed(args.seed + rank)
            torch.manual_seed(args.seed + rank)
        if rank == 0:
            _write_json(Path(args.output) / f"resolved_config_update_{start}.json",
                        {"resolved_config": config, "training_metadata": trainer.run_metadata,
                         "optimizer": ("branch_preserve_optimizer_state" if branch_mode
                                        else "exact_resume" if runtime
                                        else "fresh_finetune_schedule")})
        if start == 1:
            trainer._save_checkpoint(0)
        else:
            # Recovers an evaluation interrupted after an atomic checkpoint save.
            trainer.evaluate_bank(start - 1)
        trainer.train(args.num_updates, args.rollout_steps, start_update=start,
                      stop_at=getattr(args, "stop_at", None))
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def plot_curve(directory):
    """Plot unsmoothed fixed-bank means; never invent boundary observations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    with (directory / "learning_curve.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    references = json.loads((directory / "references.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    updates = [int(row["update"]) for row in rows]
    values = [float(row["total_time"]) if row["total_time"] else float("nan") for row in rows]
    axes[0].plot(updates, values, marker="o", ms=3, label="Greedy policy (same training instances)")
    for name, color in (("nn", "gray"), ("initial", "tab:orange")):
        if all(row["completed"] for row in references[name]):
            mean = statistics.mean(row["total_time"] for row in references[name])
            axes[0].axhline(mean, color=color, linestyle="--", label=name)
        key = f"{name}_minus_candidate_total_time"
        axes[1].plot(updates, [float(row[key]) if row.get(key) else float("nan") for row in rows],
                     label=f"{name} - candidate", color=color)
    axes[0].set_ylabel("Model-estimated flight time (s)")
    axes[1].set_ylabel("Time saved (s); positive is better")
    axes[1].set_xlabel("Fine-tuning update")
    axes[1].axhline(0, color="black", linewidth=.7)
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle("Fixed-instance overfit diagnostic: in-sample, not generalization")
    fig.tight_layout()
    fig.savefig(directory / "learning_curve.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    init = p.add_mutually_exclusive_group(required=True)
    init.add_argument("--checkpoint", help="Selected existing model; config/reward inherited unchanged.")
    init.add_argument("--from-scratch", action="store_true")
    p.add_argument("--bank", required=True)
    p.add_argument("--instance-seed", type=int, default=70000)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--fields", type=int, default=20)
    p = sub.add_parser("train")
    p.add_argument("--bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-updates", type=int, default=1000)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", help="Exact resume of this diagnostic, not its initial pretrained model.")
    p.add_argument("--branch-from", help="Checkpoint to branch from, preserving optimizer/RNG/environment.")
    p.add_argument("--actor-lr", type=float, default=None,
                   help="With --branch-from: set only the actor optimizer/LR start.")
    p.add_argument("--branch-label", default=None,
                   help="Optional label recorded in branch metadata.")
    p.add_argument("--stop-at", type=int, default=None,
                   help="Run only through this update while keeping num-updates as the target schedule.")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p = sub.add_parser("plot")
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "plot":
        plot_curve(args.output)
        return
    if args.command == "prepare":
        prepare(args)
        return
    if min(args.num_updates, args.rollout_steps, args.eval_every) < 1:
        parser.error("Update, rollout and evaluation counts must be positive.")
    if args.stop_at is not None and not (1 <= args.stop_at <= args.num_updates):
        parser.error("--stop-at must be in [1, num-updates].")
    if args.resume and args.branch_from:
        parser.error("--resume and --branch-from are different continuation modes.")
    if args.branch_from and (args.actor_lr is None or args.actor_lr <= 0):
        parser.error("--branch-from requires a positive --actor-lr.")
    if args.branch_from:
        if Path(args.branch_from).resolve().parent == Path(args.output).resolve():
            parser.error("A branch must use a new output directory, not the source run.")
    elif args.actor_lr is not None:
        parser.error("--actor-lr is only valid with --branch-from.")
    output = Path(args.output).resolve()
    if args.resume:
        if Path(args.resume).resolve().parent != output or not (output / "references.json").exists():
            parser.error("Exact resume requires the original run directory and update-0 references.")
    elif args.branch_from:
        if output.exists() and any(output.iterdir()):
            parser.error("A branch requires an empty output directory.")
    elif output.exists() and any(output.iterdir()):
        parser.error("New training requires an empty output directory; use --resume or --branch-from.")
    output.mkdir(parents=True, exist_ok=True)
    args.output = str(output)
    world_size = torch.cuda.device_count() if args.device == "cuda" else 1
    if world_size < 1:
        parser.error("No visible CUDA devices; CPU is only intended for short smoke tests.")
    if args.device == "cuda":
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)
    else:
        _worker(0, 1, args)


if __name__ == "__main__":
    main()
