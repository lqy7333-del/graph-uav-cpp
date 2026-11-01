"""Bounded sampled-vs-greedy diagnostic on the fixed 10x20 bank.

For checkpoints 0,460,580,600,780,1000:
  * reuse the existing greedy costs from evaluation/update_*.json;
  * run 16 sampled complete episodes per fixed instance (no best-of-16);
  * report mean, SE and instance-bootstrap CI;
  * align actual training-log LR/entropy/KL/clip/critic/EV;
  * summarize per-rank/per-instance transition usage.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path("/tmp/fixed-instance-overfit")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch
import torch.multiprocessing as mp

from coverage_env.env.fixed_instance_pool import load_bank
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.evaluation.maneuver_eval import run_episode
from coverage_env.rl.ppo.ppo import PPOAgent

RUN = Path("/tmp/fixed-instance-overfit/coverage_env/results/overfit20/run_seed42")
BANK = Path("/tmp/fixed-instance-overfit/coverage_env/results/overfit20/bank")
OUT = Path("/tmp/fixed-instance-overfit/coverage_env/results/overfit20/sampled_diagnostic")
CHECKPOINTS = [0, 460, 580, 600, 780, 1000]
REPLICATES = 16
SEED_BASE = 900000


def sample_choice(policy):
    def choose(environment, state):
        actions = environment.get_action_candidates()
        mask = environment.get_action_mask(actions)
        with torch.no_grad():
            output = policy(state, environment=environment,
                            candidate_actions=actions, feasible_mask=mask)
            distribution = output["action_distribution"]
            index = int(torch.multinomial(distribution, 1).item())
            return output["actions"][index], 0.0, 0.0
    return choose


def _worker(rank, world_size):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    manifest, templates = load_bank(BANK)
    rows = []
    for update in CHECKPOINTS:
        checkpoint = RUN / f"ckpt_update_{update}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = copy.deepcopy(saved["resolved_config"])
        actor = CoveragePolicy(templates[0], config=config)
        agent = PPOAgent(actor, config, device=device)
        agent.load_checkpoint(checkpoint, map_location=device,
                              restore_rng=False, load_optimizers=False)
        agent.actor.eval()
        policy = agent.actor
        choose = sample_choice(policy)
        for index in range(len(templates)):
            for replicate in range(REPLICATES):
                task = index * REPLICATES + replicate
                if task % world_size != rank:
                    continue
                seed = SEED_BASE + task
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                environment = copy.deepcopy(templates[index])
                result = run_episode(environment, choose)
                rows.append({
                    "rank": rank,
                    "update": update,
                    "index": index,
                    "seed": manifest["instances"][index]["seed"],
                    "replicate": replicate,
                    **{k: result[k] for k in (
                        "completed", "total_time", "total_distance",
                        "time_work_base", "time_nonwork_base",
                        "time_maneuver_extra", "time_service", "coverage_ratio",
                        "visits", "resupplies", "failed_steps",
                        "energy_violations", "payload_violations")},
                })
        print(f"[rank {rank}] checkpoint {update} done", flush=True)
    (OUT / f"rank_{rank}.json").write_text(
        json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def bootstrap_ci(by_instance: dict[int, list[float]], samples=4000, seed=7):
    rng = np.random.default_rng(seed)
    indices = list(by_instance)
    draws = []
    for _ in range(samples):
        chosen_instances = rng.choice(indices, size=len(indices), replace=True)
        values = []
        for index in chosen_instances:
            pool = by_instance[index]
            values.extend(rng.choice(pool, size=len(pool), replace=True).tolist())
        draws.append(float(np.mean(values)))
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Remove stale local rank files only; never touch the run directory.
    for path in OUT.glob("rank_*.json"):
        path.unlink()
    world_size = 4
    mp.spawn(_worker, args=(world_size,), nprocs=world_size, join=True)

    rows = []
    for path in sorted(OUT.glob("rank_*.json")):
        rows.extend(json.loads(path.read_text()))
    rows.sort(key=lambda r: (r["update"], r["index"], r["replicate"]))
    expected = len(CHECKPOINTS) * 10 * REPLICATES
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} sampled rows, got {len(rows)}")

    train_rows = list(csv.DictReader((RUN / "train_log.csv").open()))
    by_update = {int(r["update"]): r for r in train_rows}

    def log_slice(update):
        if update == 0:
            return None
        subset = [by_update[u] for u in range(max(1, update - 19), update + 1)]
        keys = ("actor_learning_rate", "entropy_coef", "approx_kl",
                "clip_fraction", "critic_loss", "explained_variance",
                "actor_grad_norm", "ratio_std")
        return {key: float(np.mean([float(r[key]) for r in subset])) for key in keys}

    summary = {"checkpoints": CHECKPOINTS, "replicates": REPLICATES,
               "instances": 10, "samples_per_checkpoint": 10 * REPLICATES,
               "results": {}, "training_log": {}, "rank_usage": {}}
    per_instance_rows = []
    for update in CHECKPOINTS:
        subset = [r for r in rows if r["update"] == update]
        by_instance = {}
        for r in subset:
            by_instance.setdefault(r["index"], []).append(float(r["total_time"]))
        greedy = json.loads((RUN / f"evaluation/update_{update}.json").read_text())
        greedy_by = {r["index"]: float(r["total_time"]) for r in greedy["rows"]}
        sampled_means = {i: float(np.mean(v)) for i, v in by_instance.items()}
        overall = float(np.mean(list(sampled_means.values())))
        all_values = [v for values in by_instance.values() for v in values]
        se = float(np.std(all_values, ddof=1) / math.sqrt(len(all_values)))
        low, high = bootstrap_ci(by_instance)
        greedy_mean = float(np.mean(list(greedy_by.values())))
        summary["results"][str(update)] = {
            "greedy_mean": greedy_mean,
            "sampled_mean": overall,
            "sampled_se": se,
            "sampled_ci95": [low, high],
            "sample_minus_greedy": overall - greedy_mean,
            "completed": sum(bool(r["completed"]) for r in subset),
            "samples": len(subset),
            "per_instance_sampled_mean": {str(i): sampled_means[i] for i in sorted(sampled_means)},
            "per_instance_sampled_se": {
                str(i): float(np.std(v, ddof=1) / math.sqrt(len(v)))
                for i, v in sorted(by_instance.items())},
            "per_instance_greedy": {str(i): greedy_by[i] for i in sorted(greedy_by)},
        }
        for index in sorted(by_instance):
            values = by_instance[index]
            per_instance_rows.append({
                "update": update, "index": index,
                "seed": next(r["seed"] for r in subset if r["index"] == index),
                "greedy_time": greedy_by[index],
                "sampled_mean": float(np.mean(values)),
                "sampled_std": float(np.std(values, ddof=1)),
                "sampled_se": float(np.std(values, ddof=1) / math.sqrt(len(values))),
                "sampled_min": min(values), "sampled_max": max(values),
                "completed": sum(bool(r["completed"]) for r in subset if r["index"] == index),
                "n": len(values),
            })
        summary["training_log"][str(update)] = log_slice(update)
        usage = greedy["rank_usage"]
        starts = np.sum([u["starts"] for u in usage], axis=0)
        completed = np.sum([u["completed_episodes"] for u in usage], axis=0)
        transitions = np.sum([u["transitions"] for u in usage], axis=0)
        summary["rank_usage"][str(update)] = {
            "starts": starts.tolist(), "completed_episodes": completed.tolist(),
            "transitions": transitions.tolist(),
        }

    summary["config"] = json.loads(
        (RUN / "resolved_config_update_1.json").read_text())["resolved_config"]
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT / "sampled_rows.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (OUT / "per_instance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_instance_rows[0]))
        writer.writeheader(); writer.writerows(per_instance_rows)
    print(json.dumps({u: {
        "greedy": round(v["greedy_mean"], 1),
        "sampled": round(v["sampled_mean"], 1),
        "delta": round(v["sample_minus_greedy"], 1),
        "completed": f"{v['completed']}/{v['samples']}"}
        for u, v in summary["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
