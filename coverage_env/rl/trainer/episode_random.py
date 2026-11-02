"""Random-instance (fixed 10-field size) complete-episode PPO trainer.

Each training episode is generated from a deterministic per-rank instance seed
stream.  The fixed diagnostic/validation banks are evaluation panels only and
are never sampled for training.
"""
from __future__ import annotations
import copy, csv, hashlib, json, math, random, time
from pathlib import Path
from typing import Any
import torch
import torch.distributed as dist

from coverage_env.env.instance_generator import InstanceGenerator, GeometryGenerationError
from coverage_env.env.fixed_instance_pool import load_bank
from coverage_env.rl.trainer.episode_rollout import EpisodeRolloutTrainer
from coverage_env.rl.trainer.overfit import _write_json, aggregate_rows, candidate_key
from coverage_env.rl.trainer.episode import instance_fingerprint
from coverage_env.rl.evaluation.maneuver_eval import (
    run_episode, _nn_choice, _policy_choice_factory, replay_recorded,
)
from coverage_env.rl.checkpoint import capture_rng_state, restore_rng_state


class RandomInstanceStream:
    """Deterministic, rank-disjoint episode seed stream.

    Seeds depend only on (base, rank, episode counter, generation attempt), never
    on policy actions or policy RNG consumption. Counter advances once per
    *started training episode*, including failed/truncated episodes.
    """
    BLOCK_PER_RANK = 1_000_000
    STRIDE_PER_EPISODE = 64
    MAX_ATTEMPTS = 20

    def __init__(self, base: int, rank: int, counter: int = 0, retries: int = 0):
        self.base = int(base)
        self.rank = int(rank)
        self.counter = int(counter)
        self.retries = int(retries)

    def candidate(self, counter: int | None = None, attempt: int = 0) -> int:
        c = self.counter if counter is None else int(counter)
        return (self.base + self.rank * self.BLOCK_PER_RANK
                + c * self.STRIDE_PER_EPISODE + int(attempt))

    def next_episode(self) -> int:
        counter = self.counter
        self.counter += 1
        return counter

    def state(self) -> dict:
        return {"base": self.base, "rank": self.rank, "counter": self.counter,
                "retries": self.retries, "block_per_rank": self.BLOCK_PER_RANK,
                "stride_per_episode": self.STRIDE_PER_EPISODE}

    def restore(self, state: dict) -> None:
        if int(state.get("base", -1)) != self.base or int(state.get("rank", -1)) != self.rank:
            raise ValueError("random instance stream identity changed")
        self.counter = int(state["counter"])
        self.retries = int(state.get("retries", 0))


class RandomInstanceEpisodeTrainer(EpisodeRolloutTrainer):
    """Complete episodes from freshly generated 10-field instances."""
    def __init__(self, environment, agent, *, rank=0, world_size=1,
                 checkpoint_dir=None, checkpoint_every=10, config=None,
                 diagnostic_bank=None, validation_bank=None,
                 instance_seed_base=900_000_000, num_fields=10,
                 episodes_per_rank=4, max_episode_steps=500,
                 eval_every=20):
        super().__init__(environment, agent, rank=rank, world_size=world_size,
                         checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
                         config=config, validation_bank=validation_bank,
                         episodes_per_rank=episodes_per_rank,
                         max_episode_steps=max_episode_steps)
        self.diagnostic_bank = str(Path(diagnostic_bank).resolve()) if diagnostic_bank else None
        self.validation_bank = str(Path(validation_bank).resolve()) if validation_bank else None
        self.num_fields = int(num_fields)
        self.eval_every = int(eval_every)
        if self.eval_every < 1:
            raise ValueError("eval_every must be positive")
        self.generator = InstanceGenerator(config)
        self.generator_config = copy.deepcopy(config)
        self.generator_config_sha = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
        self.stream = RandomInstanceStream(instance_seed_base, rank)
        self._episode_seq = 0
        self._episode_meta = None
        self._episode_records = []
        self._current_update = 0
        # Prevent the inherited OverfitTrainer evaluation path from using a bank.
        self.environment.directory = None

    def _new_environment(self):
        if self.num_fields != 10:
            raise ValueError("random-instance training requires exactly 10 fields")
        counter = self.stream.next_episode()
        last_error = None
        for attempt in range(RandomInstanceStream.MAX_ATTEMPTS):
            seed = self.stream.candidate(counter, attempt)
            try:
                env = self.generator.generate(
                    num_fields=self.num_fields, seed=seed,
                    randomize_each_reset=False, randomize_uav=False)
            except GeometryGenerationError as exc:
                last_error = exc
                self.stream.retries += 1
                self._record_generation_retry(seed, counter, attempt, exc)
                continue
            if len(env.fields) != self.num_fields:
                raise ValueError(f"generator returned {len(env.fields)} fields")
            fingerprint = instance_fingerprint(env)
            area = sum(float(f.area) for f in env.fields)
            self._episode_seq += 1
            self._episode_meta = {
                "update": int(self._current_update), "rank": self.rank,
                "episode_seq": self._episode_seq, "instance_seed": seed,
                "fingerprint": fingerprint, "field_count": len(env.fields),
                "area_total": area,
                "area_min": min(float(f.area) for f in env.fields),
                "area_max": max(float(f.area) for f in env.fields),
                "generation_attempts": attempt + 1,
                "generation_retries": attempt,
                "stream_counter": counter,
            }
            return env
        raise RuntimeError(f"could not generate 10-field instance after "
                           f"{RandomInstanceStream.MAX_ATTEMPTS} attempts: {last_error!r}")

    def _record_generation_retry(self, seed, counter, attempt, error):
        # Persist immediately, including attempts in an episode that never starts.
        if self.checkpoint_dir is None:
            return
        path = Path(self.checkpoint_dir) / f"generation_retries_rank_{self.rank}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        causes = []
        cause = error
        while cause is not None:
            causes.append({"type": type(cause).__name__, "message": str(cause)})
            cause = cause.__cause__
        row = {"update": self._current_update, "rank": self.rank,
               "instance_seed": seed, "stream_counter": counter,
               "attempt": attempt, "errors": causes}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _start_episode(self):
        env = self._new_environment()
        self.environment = env
        raw = self.agent.actor.module if hasattr(self.agent.actor, "module") else self.agent.actor
        if hasattr(raw, "environment"):
            raw.environment = env
        state = env.reset()
        self._note_instance()
        self._record_nn_baseline()
        self._episode_open = True
        self._episode_steps = 0
        self._episode_reward = self._episode_time = self._episode_area = 0.0
        self._episode_failed = 0
        self._last_state = state
        return state

    def _close_episode(self, success, truncated, closure_error):
        meta = dict(self._episode_meta or {})
        summary = self.environment.get_summary()
        meta.update({
            "completed": bool(success), "truncated": bool(truncated),
            "failed": bool((not success) and (not truncated)),
            "steps": int(self._episode_steps),
            "total_time": float(summary.get("total_time", 0.0)),
            "total_distance": float(summary.get("total_distance", 0.0)),
            "coverage_ratio": float(summary.get("coverage_ratio", 0.0)),
        })
        self._episode_records.append(meta)
        super()._close_episode(success, truncated, closure_error)

    def _write_episode_records(self):
        if self.checkpoint_dir is None:
            return
        path = Path(self.checkpoint_dir) / f"train_episodes_rank_{self.rank}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "update", "rank", "episode_seq", "instance_seed", "fingerprint",
                    "field_count", "area_total", "area_min", "area_max",
                    "generation_attempts", "generation_retries", "stream_counter",
                    "completed", "truncated", "failed", "steps", "total_time",
                    "total_distance", "coverage_ratio"])
                writer.writeheader()
                writer.writerows(self._episode_records)
        else:
            with path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "update", "rank", "episode_seq", "instance_seed", "fingerprint",
                    "field_count", "area_total", "area_min", "area_max",
                    "generation_attempts", "generation_retries", "stream_counter",
                    "completed", "truncated", "failed", "steps", "total_time",
                    "total_distance", "coverage_ratio"])
                writer.writerows(self._episode_records)
        self._episode_records.clear()

    def extra_rank_runtime(self):
        return {"instance_mode": "random_instances",
                "instance_stream": self.stream.state(),
                "num_fields": self.num_fields,
                "generator_config_sha": self.generator_config_sha,
                "episode_seq": self._episode_seq,
                "current_update": self._current_update}

    def _should_evaluate(self, update_idx):
        final = int(self.run_metadata.get("num_updates", -1))
        return (int(update_idx) == 0 or int(update_idx) % self.eval_every == 0
                or int(update_idx) == final)

    def _save_checkpoint(self, update_idx):
        self._write_episode_records()
        # Base save includes instance_stream via extra_rank_runtime.
        # Checkpoints can be more frequent than full panel evaluations.
        from coverage_env.rl.trainer.distributed import DistributedTrainer
        DistributedTrainer._save_checkpoint(self, update_idx)
        if self._should_evaluate(update_idx):
            self._evaluate_panel(self.diagnostic_bank, "diagnostic", update_idx)
            if self.validation_bank:
                self._evaluate_panel(self.validation_bank, "validation", update_idx)

    def _evaluate_panel(self, bank, name, update_idx):
        if not bank:
            return
        rng = capture_rng_state()
        actor = self.agent.actor.module if hasattr(self.agent.actor, "module") else self.agent.actor
        was_training = actor.training
        local = {"rows": [], "nn": [], "error": None}
        try:
            actor.eval()
            manifest, templates = load_bank(bank)
            choose = _policy_choice_factory(actor)
            for index in range(self.rank, len(templates), self.world_size):
                env = copy.deepcopy(templates[index])
                row = run_episode(env, choose, record=True)
                row.update(index=index, seed=manifest["instances"][index]["seed"])
                replay = replay_recorded(copy.deepcopy(templates[index]), row["actions"])
                same = (replay["completed"] == row["completed"]
                        and replay["termination_reason"] == row["termination_reason"]
                        and abs(replay["total_time"] - row["total_time"]) <= 1e-6)
                if not same:
                    raise RuntimeError(f"{name} replay mismatch instance {index}: "
                                       f"original=({row['completed']},{row['termination_reason']},{row['total_time']}) "
                                       f"replay=({replay['completed']},{replay['termination_reason']},{replay['total_time']})")
                row["replay_ok"] = True; row["replay"] = replay
                local["rows"].append(row)
                if update_idx == 0:
                    base = run_episode(copy.deepcopy(templates[index]), _nn_choice, record=True)
                    base.update(index=index, seed=row["seed"])
                    local["nn"].append(base)
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
        if any(g["error"] for g in gathered):
            raise RuntimeError(f"{name} evaluation failed: {[g['error'] for g in gathered]}")
        if self.rank != 0:
            return
        directory = Path(self.checkpoint_dir) / name
        directory.mkdir(parents=True, exist_ok=True)
        rows = sorted([r for g in gathered for r in g["rows"]], key=lambda r: r["index"])
        expected = list(range(len(templates)))
        if [r["index"] for r in rows] != expected:
            raise RuntimeError(f"{name} evaluation did not cover the panel")
        if update_idx == 0:
            refs = {"initial": rows, "nn": sorted(
                [r for g in gathered for r in g["nn"]], key=lambda r: r["index"])}
            _write_json(directory / "references.json", refs)
        else:
            refs = json.loads((directory / "references.json").read_text(encoding="utf-8"))
        summary = aggregate_rows(rows, refs)
        summary.update(update=update_idx, scope=f"fixed_{name}_panel")
        _write_json(directory / "evaluation" / f"update_{update_idx}.json",
                    {"summary": summary, "rows": rows})
        history = [json.loads(p.read_text(encoding="utf-8"))["summary"]
                   for p in (directory / "evaluation").glob("update_*.json")]
        history.sort(key=lambda r: r["update"])
        _write_json(directory / "best.json", min(history, key=candidate_key))
        keys = sorted(set().union(*(r.keys() for r in history)))
        with (directory / f"{name}_curve.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(history)
        print(f"[{name}] update={update_idx} complete={summary['completed']}/"
              f"{summary['instances']} time={summary['total_time']}", flush=True)

    def train(self, num_updates, rollout_steps=None, *, start_update=1, stop_at=None):
        self.run_metadata.update({
            "collection_mode": "complete_episodes",
            "instance_mode": "random_instances",
            "instance_seed_base": self.stream.base,
            "num_fields": self.num_fields,
            "generator_config_sha": self.generator_config_sha,
            "diagnostic_bank": self.diagnostic_bank,
            "validation_bank": self.validation_bank,
            "eval_every": self.eval_every,
        })
        super().train(num_updates, rollout_steps, start_update=start_update, stop_at=stop_at)
        self._write_episode_records()
