"""Single-training, multi-GPU Distributed Data Parallel (DDP) trainer.

Each rank runs its own environment (different seed) and collects a fixed-length
rollout. The model is wrapped in :class:`DistributedDataParallel` so gradients
are all-reduced across ranks, keeping a single shared set of parameters.

The fixed rollout length is what makes DDP viable here: the existing
``Trainer`` updates once per *complete episode*, and episode length varies with
the sampled fields. DDP requires every rank to issue the same number of
backward passes, so we collect exactly ``rollout_steps`` transitions per update
and reset the environment whenever an episode ends.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import math
import os
import socket
import json
import random
import time
from copy import deepcopy
from pathlib import Path
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from coverage_env.config import load_config
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.buffer.rollout_buffer import RolloutBuffer
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.trainer import _clone_graph
from coverage_env.rl.trainer.episode import nearest_neighbor_return, current_potential, learning_reward
from coverage_env.rl.checkpoint import (
    capture_rng_state,
    parameter_sha256,
    provenance,
    restore_rng_state,
    assert_rank_parameters_equal,
)


# Correctness-gate tolerances.  The absolute term dominates near zero; the
# relative term scales with the value magnitude.  They are deliberately tight
# enough to catch the heading/next-state class of error (observed gap 150+)
# while leaving float32 and CUDA-reduction noise (observed ~1e-6) untouched.
VALUE_GAP_ABS = 1e-2
VALUE_GAP_REL = 1e-4
TAIL_VALUE_ABS = 1e-2
TAIL_VALUE_REL = 1e-4
REPLAY_LOGPROB_ABS = 1e-4
REPLAY_VALUE_ABS = 1e-2
REWARD_CLOSURE_ABS = 1e-6


def _free_port() -> int:
    """Return an available TCP port for the NCCL rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class DistributedTrainer:
    """Collect a fixed-length rollout per rank, then one synchronized PPO update."""

    def __init__(self, environment, agent, *, rank: int = 0, world_size: int = 1,
                 checkpoint_dir=None, checkpoint_every: int = 50,
                 config: dict | None = None) -> None:
        self.environment = environment
        self.agent = agent
        # Full experiment config; the single-process Trainer stores the same thing.
        # Read defensively (tests build this class without a config).
        self.config = config or {}
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.buffer = RolloutBuffer()
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = int(checkpoint_every)
        self._count_emas = {}
        self.run_metadata = {}
        self._rank_stats = self._empty_rank_stats()
        self._geometry_signatures: set = set()
        # Toy environments in unit tests do not expose UAV/field state; the
        # correctness gate is for the real training environment.
        self.value_gate_enabled = (
            bool(self.config.get("ppo", {}).get("value_gate", True))
            and bool(getattr(environment, "uavs", None))
        )

    @staticmethod
    def _empty_rank_stats() -> dict:
        """Per-rank counters used by the run summary and exact-resume payload."""
        return {
            "transitions": 0,
            "episodes": 0,
            "optimizer_steps": 0,
            "updates": 0,
            "wall_seconds": 0.0,
            "instances_drawn": 0,
            "distinct_geometries": 0,
            "field_counts": {},
        }

    def _note_instance(self) -> None:
        """Record the instance shape drawn by the latest reset.

        Only the distribution of field counts and the number of distinct field
        geometries are stored; this is provenance for the short-chain gate, not
        a training signal.  A geometry signature uses field ids and rounded areas
        so it is stable across deep copies and serialization.
        """
        signature = tuple(sorted(
            (str(getattr(field, "field_id", index)),
             round(float(getattr(field, "area", 0.0)), 3))
            for index, field in enumerate(getattr(self.environment, "fields", []) or [])
        ))
        if signature not in self._geometry_signatures:
            self._geometry_signatures.add(signature)
            self._rank_stats["distinct_geometries"] = len(self._geometry_signatures)
        self._rank_stats["instances_drawn"] = int(
            self._rank_stats.get("instances_drawn", 0)
        ) + 1

    def _all_reduce_max(self, value: float) -> float:
        """Return the maximum scalar across ranks (no-op when not distributed)."""
        tensor = torch.tensor([float(value)], dtype=torch.float64,
                              device=self.agent.device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def _all_reduce_min(self, value: float) -> float:
        """Return the minimum scalar across ranks (no-op when not distributed)."""
        tensor = torch.tensor([float(value)], dtype=torch.float64,
                              device=self.agent.device)
        if dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return float(tensor.item())

    @staticmethod
    def _state_digest(observation) -> str:
        payload = json.dumps(observation, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalized_error(error: float, left: float, right: float,
                          abs_tol: float, rel_tol: float) -> float:
        """Per-pair relative tolerance, rather than one global value scale."""
        scale = max(abs(float(left)), abs(float(right)), 1.0)
        return abs(float(error)) / (abs_tol + rel_tol * scale)

    def _local_gate_metrics(self, transitions, closure_error_max: float) -> dict:
        """Compute one rank's pre-update correctness metrics.

        All comparisons use the pair's own value scale.  A single large value
        elsewhere in the rollout must not loosen the tolerance for a small one.
        """
        rewards = [float(t.reward) for t in transitions]
        values = [float(t.value) for t in transitions]
        next_values = [float(t.next_value) for t in transitions]
        finite = all(math.isfinite(x) for x in rewards + values + next_values)
        pair_errors = []
        pair_q = []
        pair_indices = []
        gap_index = 0
        for i in range(len(transitions) - 1):
            if transitions[i].done:
                continue
            error = abs(next_values[i] - values[i + 1])
            pair_errors.append(error)
            pair_indices.append(i)
            pair_q.append(self._normalized_error(
                error, next_values[i], values[i + 1],
                VALUE_GAP_ABS, VALUE_GAP_REL))
            if not math.isfinite(error):
                finite = False
        if pair_q:
            gap_index = pair_indices[
                max(range(len(pair_q)), key=lambda i: pair_q[i])]
        gap_abs_index = pair_indices[
            max(range(len(pair_errors)), key=lambda i: pair_errors[i])] \
            if pair_errors else 0
        tail_error = 0.0
        tail_q = 0.0
        tail_expected = 0.0
        if transitions:
            last = transitions[-1]
            if last.done:
                tail_error = abs(float(last.next_value))
                tail_q = self._normalized_error(
                    tail_error, 0.0, float(last.next_value),
                    TAIL_VALUE_ABS, TAIL_VALUE_REL)
            else:
                actor = (self.agent.actor.module
                         if hasattr(self.agent.actor, "module")
                         else self.agent.actor)
                _embeddings, _embedding, critic_input = actor.encode_graph(
                    last.next_state["graph"], environment=self.environment,
                    state=last.next_state["observation"],
                )
                tail_expected = float(self.agent.critic(critic_input).reshape(()).item())
                tail_error = abs(tail_expected - float(last.next_value))
                tail_q = self._normalized_error(
                    tail_error, tail_expected, float(last.next_value),
                    TAIL_VALUE_ABS, TAIL_VALUE_REL)
                if not math.isfinite(tail_expected):
                    finite = False
        replay_logprob_error = 0.0
        replay_value_error = 0.0
        replay_bad_indices = []
        if transitions:
            indices = sorted(set([0, len(transitions) // 3,
                                   2 * len(transitions) // 3, len(transitions) - 1]))
            try:
                with torch.no_grad():
                    log_probs, _entropy, replay_values = (
                        self.agent._per_transition_minibatch_rows(transitions, indices)
                    )
                if not bool(torch.isfinite(log_probs).all().item()):
                    finite = False
                if not bool(torch.isfinite(replay_values).all().item()):
                    finite = False
                lp_errors = []
                val_errors = []
                for j, index in enumerate(indices):
                    lp = float(log_probs[j].item())
                    val = float(replay_values[j].item())
                    lp_error = abs(lp - float(transitions[index].log_prob))
                    val_error = abs(val - float(transitions[index].value))
                    if (not math.isfinite(lp_error) or
                            not math.isfinite(val_error) or
                            lp_error > REPLAY_LOGPROB_ABS or
                            val_error > REPLAY_VALUE_ABS):
                        replay_bad_indices.append(index)
                    lp_errors.append(lp_error)
                    val_errors.append(val_error)
                replay_logprob_error = max(lp_errors) if lp_errors else 0.0
                replay_value_error = max(val_errors) if val_errors else 0.0
                if not (math.isfinite(replay_logprob_error) and
                        math.isfinite(replay_value_error)):
                    finite = False
            except Exception:
                # A replay failure is a correctness failure, not a training signal.
                finite = False
                replay_logprob_error = float("inf")
                replay_value_error = float("inf")
        metrics = {
            "value_state_gap_max": max(pair_errors) if pair_errors else 0.0,
            "value_state_scale": max(
                [abs(v) for v in values] + [abs(v) for v in next_values] + [1.0]),
            "pair_q_max": max(pair_q) if pair_q else 0.0,
            "tail_value_error": tail_error,
            "tail_value_scale": max(abs(float(transitions[-1].value)),
                                    abs(float(transitions[-1].next_value)), 1.0)
            if transitions else 1.0,
            "tail_q": tail_q,
            "replay_logprob_error": replay_logprob_error,
            "replay_value_error": replay_value_error,
            "episode_closure_error_max": float(closure_error_max),
            "finite": bool(finite),
            "gap_index": int(gap_index),
            "gap_abs_index": int(gap_abs_index),
            "replay_bad_indices": replay_bad_indices,
            "tail_index": len(transitions) - 1 if transitions else 0,
        }
        return metrics

    def _write_gate_failure(self, update_idx: int, reason: str, local: dict,
                            global_metrics: dict) -> None:
        """Persist the samples that actually triggered each failed check.

        Diagnostic writes must never prevent the all-rank barrier/abort path.
        All writes are wrapped; the barrier is in ``finally``.
        """
        directory = None
        try:
            if self.checkpoint_dir is not None:
                directory = Path(self.checkpoint_dir) / "gate_failures"
                directory.mkdir(parents=True, exist_ok=True)
            transitions = list(getattr(self, "buffer", []).transitions)
            bad = set()
            for key in ("gap_index", "gap_abs_index"):
                index = int(local.get(key, 0) or 0)
                if transitions:
                    bad.add(min(index, len(transitions) - 1))
                    bad.add(min(index + 1, len(transitions) - 1))
            for index in local.get("replay_bad_indices", []) or []:
                if transitions:
                    bad.add(min(int(index), len(transitions) - 1))
            if transitions:
                bad.add(min(int(local.get("tail_index", len(transitions) - 1)),
                            len(transitions) - 1))
            if not bad and transitions:
                bad.add(0)
                bad.add(len(transitions) - 1)
            bad = sorted(bad)[:16]
            samples = {}
            for index in bad:
                if not (0 <= index < len(transitions)):
                    continue
                transition = transitions[index]
                samples[index] = {
                    "state": transition.state,
                    "next_state": transition.next_state,
                    "action": transition.action,
                    "reward": float(transition.reward),
                    "value": float(transition.value),
                    "next_value": float(transition.next_value),
                    "done": bool(transition.done),
                    "truncated": bool(getattr(transition, "truncated", False)),
                }
        except Exception:
            transitions = []
            bad = []
            samples = {}
        try:
            if directory is not None:
                prefix = f"update_{update_idx}_rank_{self.rank}"
                payload = {
                    "sha": provenance().get("revision"),
                    "rank": self.rank,
                    "update": update_idx,
                    "reason": reason,
                    "thresholds": {
                        "value_gap_abs": VALUE_GAP_ABS,
                        "value_gap_rel": VALUE_GAP_REL,
                        "tail_abs": TAIL_VALUE_ABS,
                        "tail_rel": TAIL_VALUE_REL,
                        "replay_logprob_abs": REPLAY_LOGPROB_ABS,
                        "replay_value_abs": REPLAY_VALUE_ABS,
                        "closure_abs": REWARD_CLOSURE_ABS,
                    },
                    "local": local,
                    "global": global_metrics,
                    "run_metadata": self.run_metadata,
                    "config": self.config,
                    "bad_indices": bad,
                    "transitions": [
                        {
                            "index": index,
                            "reward": samples[index]["reward"],
                            "value": samples[index]["value"],
                            "next_value": samples[index]["next_value"],
                            "done": samples[index]["done"],
                            "truncated": samples[index]["truncated"],
                            "state_digest": self._state_digest(
                                samples[index]["state"].get("observation")),
                            "next_state_digest": self._state_digest(
                                samples[index]["next_state"].get("observation")),
                        }
                        for index in bad if index in samples
                    ],
                }
                (directory / f"{prefix}.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
                torch.save({"samples": samples},
                           directory / f"{prefix}.pt")
        except Exception:
            pass
        finally:
            if dist.is_initialized():
                dist.barrier()
        if self.rank == 0 and directory is not None:
            try:
                summary = {
                    "sha": provenance().get("revision"),
                    "update": update_idx,
                    "reason": reason,
                    "thresholds": {
                        "value_gap_abs": VALUE_GAP_ABS,
                        "value_gap_rel": VALUE_GAP_REL,
                        "tail_abs": TAIL_VALUE_ABS,
                        "tail_rel": TAIL_VALUE_REL,
                        "replay_logprob_abs": REPLAY_LOGPROB_ABS,
                        "replay_value_abs": REPLAY_VALUE_ABS,
                        "closure_abs": REWARD_CLOSURE_ABS,
                    },
                    "local": local,
                    "global": global_metrics,
                    "files": sorted(p.name for p in directory.glob(
                        f"update_{update_idx}_rank_*")),
                }
                (directory / f"update_{update_idx}_summary.json").write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
            except Exception:
                pass

    def _gate_or_abort(self, local: dict, update_idx: int) -> dict:
        """All-reduce local correctness metrics and abort every rank on failure."""
        keys = (
            "value_state_gap_max", "value_state_scale", "pair_q_max",
            "tail_value_error", "tail_value_scale", "tail_q",
            "replay_logprob_error", "replay_value_error",
            "episode_closure_error_max",
        )
        global_metrics = {key: self._all_reduce_max(local.get(key, 0.0))
                          for key in keys}
        metrics_finite = all(
            math.isfinite(float(local.get(key, 0.0))) for key in keys)
        all_finite = bool(self._all_reduce_min(
            1.0 if (local.get("finite") and metrics_finite) else 0.0))
        failures = []
        if not all_finite:
            failures.append("nonfinite buffer/state/replay metric")
        if global_metrics["pair_q_max"] > 1.0:
            failures.append(
                f"pair_q_max {global_metrics['pair_q_max']:.3e} > 1 "
                f"(abs gap {global_metrics['value_state_gap_max']:.3e})")
        if global_metrics["tail_q"] > 1.0:
            failures.append(
                f"tail_q {global_metrics['tail_q']:.3e} > 1 "
                f"(abs tail error {global_metrics['tail_value_error']:.3e})")
        if global_metrics["replay_logprob_error"] > REPLAY_LOGPROB_ABS:
            failures.append(
                f"replay_logprob_error {global_metrics['replay_logprob_error']:.3e}")
        if global_metrics["replay_value_error"] > REPLAY_VALUE_ABS:
            failures.append(
                f"replay_value_error {global_metrics['replay_value_error']:.3e}")
        if global_metrics["episode_closure_error_max"] > REWARD_CLOSURE_ABS:
            failures.append(
                f"episode_closure_error_max {global_metrics['episode_closure_error_max']:.3e}")
        if failures:
            reason = "; ".join(failures)
            self._write_gate_failure(update_idx, reason, local, global_metrics)
            raise RuntimeError(f"pre-update correctness gate failed at update "
                               f"{update_idx} on rank {self.rank}: {reason}")
        return global_metrics

    def _post_update_gate(self, stats: dict, update_idx: int) -> None:
        """Reject NaN/Inf produced by the optimizer before saving a checkpoint."""
        keys = ("actor_loss", "critic_loss", "entropy", "policy_loss",
                "actor_grad_norm", "approx_kl")
        finite = all(math.isfinite(float(stats.get(key, 0.0))) for key in keys)
        for module in (self.agent.actor, self.agent.critic):
            source = module.module if hasattr(module, "module") else module
            for parameter in source.parameters():
                if not bool(torch.isfinite(parameter).all().item()):
                    finite = False
                    break
        if not self._all_reduce_min(1.0 if finite else 0.0):
            local = {"finite": False, "stats": stats, "update": update_idx}
            self._write_gate_failure(update_idx, "nonfinite post-update parameters/stats",
                                     local, {})
            raise RuntimeError(
                f"post-update numerical gate failed at update {update_idx} "
                f"on rank {self.rank}")

    def write_run_summary(self) -> None:
        """Gather per-rank throughput counters and write ``run_summary.json``."""
        if self.checkpoint_dir is None:
            return
        local = deepcopy(self._rank_stats)
        gathered = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if self.rank != 0:
            return
        merged_counts: dict = {}
        for stats in gathered:
            for count, number in (stats or {}).get("field_counts", {}).items():
                merged_counts[str(count)] = merged_counts.get(str(count), 0) + int(number)
        per_rank = [dict(stats or {}, rank=index) for index, stats in enumerate(gathered)]
        global_stats = {
            "transitions": sum(int(s.get("transitions", 0)) for s in gathered),
            "episodes": sum(int(s.get("episodes", 0)) for s in gathered),
            # All ranks issue the same number of synchronized optimizer steps.
            "optimizer_steps_global_synchronous": max(
                (int(s.get("optimizer_steps", 0)) for s in gathered), default=0
            ),
            "optimizer_steps_per_rank_sum": sum(
                int(s.get("optimizer_steps", 0)) for s in gathered
            ),
            "wall_seconds_max": max(
                (float(s.get("wall_seconds", 0.0)) for s in gathered), default=0.0
            ),
            "field_counts": merged_counts,
        }
        payload = {
            "training_metadata": deepcopy(self.run_metadata),
            "per_rank": per_rank,
            "global": global_stats,
        }
        path = Path(self.checkpoint_dir) / "run_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        print(f"[rank 0] wrote run summary -> {path}", flush=True)

    def train(self, num_updates: int, rollout_steps: int, *, start_update: int = 1) -> None:
        """Run ``num_updates`` synchronized updates of ``rollout_steps`` steps each.

        ``start_update`` lets a resumed run continue at the next update index
        while still using the same absolute target (``num_updates``).
        """
        # Per-field-count reward EMAs, so the log can report a mean in which every
        # field count carries the same weight. The plain mean is biased by whichever
        # counts this update happened to draw (5-field and 20-field episodes differ by
        # ~4 reward points), which is why the raw curve was unreadable.
        count_emas = self._count_emas
        ema_alpha = 0.1
        self.run_metadata.update(num_updates=num_updates, rollout_steps=rollout_steps,
                                 world_size=self.world_size)

        for update_idx in range(max(1, start_update), num_updates + 1):
            update_started = time.perf_counter()
            # Sampling is stochastic and must not feed the autograd graph.
            self.agent.actor.eval()
            state = self.environment.reset()
            self._note_instance()
            self._record_nn_baseline()
            rollout_reward = 0.0
            # Under field-count generalization a 128-step rollout spans several
            # episodes with different field counts, so the rollout sum alone is
            # dominated by instance variance. Accumulate per-episode returns and
            # label each with the episode's own field count so the log can show
            # a trend and compare magnitudes across field counts.
            episode_reward = 0.0
            episode_time = 0.0
            episode_area = 0.0
            episode_failed = 0
            closure_error_max = 0.0
            reward_mode = str(self.config.get("rl", {}).get(
                "reward", {}).get("mode", "distance")).lower()
            shaped_rollout_reward = 0.0
            completed_rewards: list[float] = []
            completed_fields: list[int] = []
            completed_summaries = []

            for _ in range(rollout_steps):
                with torch.no_grad():
                    output = self.agent.actor(state, environment=self.environment)
                    distribution = output["action_distribution"]
                    retries = 0
                    while distribution.numel() == 0 or float(distribution.detach().sum()) <= 0.0:
                        if self.buffer.transitions and not self.buffer.transitions[-1].done:
                            raise RuntimeError("Nonterminal environment returned no feasible action.")
                        retries += 1
                        if retries > 16:
                            raise RuntimeError("Could not sample a feasible initial instance after 16 resets.")
                        state = self.environment.reset()
                        self._note_instance()
                        self._record_nn_baseline()
                        output = self.agent.actor(state, environment=self.environment)
                        distribution = output["action_distribution"]
                    action_index = int(torch.multinomial(distribution, 1).item())
                    action = output["actions"][action_index]
                    log_prob = torch.log(distribution[action_index].clamp_min(1e-12))
                    value = self.agent.value_from_output(output)

                before = current_potential(self.environment, self.config)
                next_state, reward, env_done, _info = self.environment.step(action)
                # Potential-based shaping (see coverage_env/rl/shaping.py). The
                # policy learns from `learning_reward`; everything logged as
                # `reward` stays raw so runs remain comparable and evaluation is
                # unaffected.
                shaped = learning_reward(self.environment, reward, before, env_done,
                                         self.config, self.agent.gamma)
                shaped_rollout_reward += float(shaped)
                rollout_reward += float(reward)
                episode_reward += float(reward)
                episode_time += float(_info.get("time_total", 0.0) or 0.0)
                episode_area += float(_info.get("coverage_gain", 0.0) or 0.0)
                episode_failed += int(bool(_info.get("failed")))
                if env_done:
                    if reward_mode == "time":
                        reward_cfg = self.config.get("rl", {}).get("reward", {})
                        width = float(reward_cfg.get("coverage_width", 10.0))
                        nominal = float(reward_cfg.get("nominal_work_speed", 8.0))
                        time_scale = float(reward_cfg.get("time_scale", 100.0))
                        failure_penalty = float(self.config.get("rl", {}).get(
                            "failure_penalty", -1.0))
                        expected = -(
                            episode_time - episode_area / (width * nominal)
                        ) / time_scale + failure_penalty * episode_failed
                        closure_error_max = max(
                            closure_error_max, abs(episode_reward - expected))
                    episode_time = episode_area = 0.0
                    episode_failed = 0
                    # The environment still describes the episode that just
                    # finished, so its field count labels this episode.
                    completed_rewards.append(episode_reward)
                    self._rank_stats["episodes"] += 1
                    count_here = len(self.environment.fields)
                    completed_fields.append(count_here)
                    _counts = self._rank_stats["field_counts"]
                    _counts[str(count_here)] = _counts.get(str(count_here), 0) + 1
                    completed_summaries.append(self.environment.get_summary())
                    previous = count_emas.get(count_here)
                    count_emas[count_here] = (
                        episode_reward if previous is None
                        else previous + ema_alpha * (episode_reward - previous)
                    )
                    episode_reward = 0.0
                with torch.no_grad():
                    # DDP does not proxy custom attributes; reach the wrapped
                    # CoveragePolicy via ``.module`` for graph construction and
                    # encoding (``value_for_graph`` also assumes the raw actor).
                    next_graph = self.agent.actor.module.graph_builder.build_graph(self.environment)
                    if env_done:
                        next_value = 0.0
                    else:
                        _embeddings, _global_embedding, critic_input = (
                            self.agent.actor.module.encode_graph(
                                next_graph, environment=self.environment,
                                state=next_state,
                            )
                        )
                        next_value = self.agent.critic(critic_input).reshape(()).item()

                self.buffer.add(
                    {
                        "observation": deepcopy(state),
                        "graph": _clone_graph(output["graph"]),
                        "actions": list(output["actions"]),
                        "mask": output["mask"].detach().cpu().tolist(),
                    },
                    action,
                    shaped,
                    {
                        "observation": deepcopy(next_state),
                        "graph": _clone_graph(next_graph),
                    },
                    env_done,
                    log_prob.detach().item(),
                    value.detach().item(),
                    next_value,
                )
                self._rank_stats["transitions"] += 1
                state = self.environment.reset() if env_done else next_state
                if env_done:
                    self._note_instance()
                # Only a finished episode starts a new instance; recording on every
                # step re-ran a full nearest-neighbour rollout per transition
                # (18 min/update instead of ~3).
                if env_done:
                    self._record_nn_baseline()

            # This tail is a time-limit fragment, not a complete broadcast label.
            # GAE still bootstraps its stored next_value.
            if self.buffer.transitions and not self.buffer.transitions[-1].done:
                self.buffer.transitions[-1].truncated = True

            # Keep the SAME deterministic network mode used for sampling. Calling
            # ``train()`` here re-enabled dropout, so the recomputed log-probs differed
            # from the sampled ones even before any weight update: the PPO ratio left 1
            # (measured 0.56-1.70, 36% outside the clip band). DDP reduces gradients
            # regardless of ``module.training``, and ``eval()`` does not disable
            # autograd, so nothing about training requires ``train()`` here.
            self.agent.actor.eval()
            self.agent.critic.eval()
            # With a fixed parameter set and eval mode, the stored bootstrap value
            # for a nonterminal transition must equal the next transition's value
            # under the same state features.  This is a target-correctness metric,
            # not a reward; a large value means the next-state observation was not
            # encoded consistently.
            _transitions = self.buffer.transitions
            _gaps = [
                abs(float(t.next_value) - float(_transitions[i + 1].value))
                for i, t in enumerate(_transitions[:-1])
                if not t.done
            ]
            local_gate = {
                "value_state_gap_max": max(_gaps) if _gaps else 0.0,
                "value_state_scale": max(
                    [abs(float(t.value)) for t in _transitions]
                    + [abs(float(t.next_value)) for t in _transitions] + [1.0]),
                "pair_q_max": 0.0,
                "tail_value_error": 0.0,
                "tail_value_scale": 1.0,
                "tail_q": 0.0,
                "replay_logprob_error": 0.0,
                "replay_value_error": 0.0,
                "episode_closure_error_max": closure_error_max,
                "finite": all(math.isfinite(float(t.reward))
                              and math.isfinite(float(t.value))
                              and math.isfinite(float(t.next_value))
                              for t in _transitions),
            }
            if self.value_gate_enabled:
                try:
                    local_gate = self._local_gate_metrics(
                        _transitions, closure_error_max)
                except Exception as exc:
                    local_gate = {
                        "value_state_gap_max": float("inf"),
                        "value_state_scale": 1.0,
                        "pair_q_max": float("inf"),
                        "tail_value_error": float("inf"),
                        "tail_value_scale": 1.0,
                        "tail_q": float("inf"),
                        "replay_logprob_error": float("inf"),
                        "replay_value_error": float("inf"),
                        "episode_closure_error_max": closure_error_max,
                        "finite": False,
                        "error": repr(exc),
                    }
                gate_metrics = self._gate_or_abort(local_gate, update_idx)
                value_state_gap_max = gate_metrics["value_state_gap_max"]
                pair_q_max = gate_metrics["pair_q_max"]
                tail_value_error = gate_metrics["tail_value_error"]
                tail_q = gate_metrics["tail_q"]
                replay_logprob_error = gate_metrics["replay_logprob_error"]
                replay_value_error = gate_metrics["replay_value_error"]
                episode_closure_global = gate_metrics["episode_closure_error_max"]
            else:
                gate_metrics = local_gate
                value_state_gap_max = local_gate["value_state_gap_max"]
                pair_q_max = 0.0
                tail_value_error = 0.0
                tail_q = 0.0
                replay_logprob_error = 0.0
                replay_value_error = 0.0
                episode_closure_global = closure_error_max
            # Advance entropy annealing before the update so the logged value is
            # the one actually used.
            progress = self.agent.set_training_progress(
                (update_idx - 1) / max(1, num_updates - 1)
            )
            current_entropy_coef = progress["entropy_coef"]
            stats = self.agent.update(self.buffer)
            if self.value_gate_enabled:
                self._post_update_gate(stats, update_idx)
            self.buffer.clear()
            self._rank_stats["optimizer_steps"] += int(stats.get("updates", 0))
            self._rank_stats["updates"] += 1
            self._rank_stats["wall_seconds"] += time.perf_counter() - update_started

            if self.rank == 0:
                summary = self.environment.get_summary()
                episodes = len(completed_rewards)
                mean_episode_reward = (
                    sum(completed_rewards) / episodes if episodes else None
                )
                mean_episode_fields = (
                    sum(completed_fields) / episodes if episodes else None
                )
                row = {
                    "update": update_idx,
                    "reward": round(rollout_reward, 4),
                    "actor_loss": round(stats["actor_loss"], 6),
                    "shaped_reward": round(shaped_rollout_reward, 4),
                    "entropy_coef": round(float(current_entropy_coef), 6),
                    "actor_learning_rate": round(
                        float(progress["actor_learning_rate"]), 8
                    ),
                    "critic_loss": round(stats["critic_loss"], 6),
                    # Completed-episode means on rank 0, not a partial/reset mission.
                    "coverage_ratio": (round(sum(s["coverage_ratio"] for s in completed_summaries) / episodes, 4)
                                       if episodes else ""),
                    "total_distance": (round(sum(s["total_distance"] for s in completed_summaries) / episodes, 1)
                                       if episodes else ""),
                    "metrics_scope": "rank0_completed_episodes",
                    "current_partial_distance": round(summary["total_distance"], 1),
                    "current_partial_coverage": round(summary["coverage_ratio"], 4),
                    "episodes": episodes,
                    "transitions_this_update": rollout_steps,
                    "transitions_total_rank0": self._rank_stats["transitions"],
                    "episodes_total_rank0": self._rank_stats["episodes"],
                    "optimizer_steps_total": self._rank_stats["optimizer_steps"],
                    "update_wall_seconds": round(
                        float(self._rank_stats["wall_seconds"]), 3
                    ),
                    "balanced_reward": (
                        round(sum(count_emas.values()) / len(count_emas), 4)
                        if count_emas else ""
                    ),
                    "mean_ep_reward": (
                        round(mean_episode_reward, 4)
                        if mean_episode_reward is not None else ""
                    ),
                    "mean_ep_fields": (
                        round(mean_episode_fields, 2)
                        if mean_episode_fields is not None else ""
                    ),
                    # Diagnostics for "why doesn't the reward improve?":
                    # `entropy`/`policy_loss` separate the entropy bonus from the
                    # advantage-weighted surrogate, `actor_grad_norm` shows how
                    # much gradient actually reaches the actor, and
                    # `explained_variance` shows whether the critic explains
                    # enough return variance for the advantages to be useful.
                    "entropy": round(stats["entropy"], 5),
                    "policy_loss": round(stats["policy_loss"], 6),
                    "advantage_std": round(stats["advantage_std"], 5),
                    "return_mean": round(stats["return_mean"], 4),
                    "return_std": round(stats["return_std"], 4),
                    "value_mean": round(stats["value_mean"], 4),
                    "value_std": round(stats["value_std"], 4),
                    "explained_variance": round(stats["explained_variance"], 5),
                    "ratio_std": round(stats["ratio_std"], 6),
                    "clip_fraction": round(stats["clip_fraction"], 5),
                    "approx_kl": round(stats["approx_kl"], 7),
                    "actor_grad_norm": round(stats["actor_grad_norm"], 5),
                    "value_state_gap_max": round(value_state_gap_max, 8),
                    "pair_q_max": round(pair_q_max, 8),
                    "tail_value_error": round(tail_value_error, 8),
                    "tail_q": round(tail_q, 8),
                    "replay_logprob_error": round(replay_logprob_error, 8),
                    "replay_value_error": round(replay_value_error, 8),
                    "episode_closure_error_max": round(
                        episode_closure_global, 12),
                    "value_gate_passed": True,
                }
                episode_note = (
                    f" | eps={episodes} mean_ep_reward={mean_episode_reward:.2f} "
                    f"mean_ep_fields={mean_episode_fields:.1f}"
                    if episodes else " | eps=0"
                )
                print(
                    f"[DDP] update {update_idx}/{num_updates} "
                    f'adv={getattr(self.agent, "last_advantage_mode", "gae")} | '
                    f"steps={rollout_steps} "
                    f"reward={rollout_reward:.3f} current_coverage={summary['coverage_ratio']:.1%} "
                    f"actor_loss={stats['actor_loss']:.5f} critic_loss={stats['critic_loss']:.5f}"
                    f"{episode_note}\n"
                    f"      entropy={stats['entropy']:.3f} policy_loss={stats['policy_loss']:+.5f} "
                    f"explained_var={stats['explained_variance']:+.3f} "
                    f"grad_norm={stats['actor_grad_norm']:.4f} "
                    f"ratio_std={stats['ratio_std']:.4f} clip_frac={stats['clip_fraction']:.3f} "
                    f"adv_std={stats['advantage_std']:.3f} return_std={stats['return_std']:.3f} "
                    f"value={stats['value_mean']:+.2f}±{stats['value_std']:.2f}",
                    flush=True,
                )
                self._append_log(row)
            if self.checkpoint_dir and (update_idx % self.checkpoint_every == 0 or update_idx == num_updates):
                self._save_checkpoint(update_idx)

        self.write_run_summary()

    def _record_nn_baseline(self):
        """Nearest-neighbour return for the current instance (baseline-replacement test)."""
        mode = str(self.agent.config.get("advantage_mode", "gae")).lower()
        if mode != "broadcast_nn":
            return
        total, fingerprint = nearest_neighbor_return(
            self.environment, self.config, self.agent.gamma)
        self.last_baseline_fingerprint = fingerprint
        self.agent.nn_baselines.append(total)


    def _append_log(self, row: dict) -> None:
        """Append one training row to the CSV log (rank 0 only)."""
        if self.checkpoint_dir is None:
            return
        from pathlib import Path

        log_path = Path(self.checkpoint_dir) / "train_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with log_path.open("a", encoding="utf-8") as stream:
            if write_header:
                stream.write(",".join(row.keys()) + "\n")
            stream.write(",".join(str(value) for value in row.values()) + "\n")

    def _save_checkpoint(self, update_idx: int) -> None:
        """Save actor/critic/optimizer state without the DDP ``module.`` prefix."""
        if self.checkpoint_dir is None:
            return
        assert_rank_parameters_equal(self.agent.actor, self.agent.critic)
        local = {"rng": capture_rng_state(), "environment": deepcopy(self.environment),
                 "count_emas": deepcopy(self._count_emas),
                 "rank_stats": deepcopy(self._rank_stats)}
        states = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(states, local)
        else:
            states = [local]
        if self.rank != 0:
            return
        path = Path(self.checkpoint_dir) / f"ckpt_update_{update_idx}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = self.agent.checkpoint_payload()
        checkpoint.update(update_idx=update_idx, rank_runtime=states,
                          training_metadata=deepcopy(self.run_metadata))
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)
        print(f"[rank 0] saved checkpoint -> {path}", flush=True)


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    """One distributed process bound to GPU ``rank``."""
    dist.init_process_group(
        "nccl", init_method="env://", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(minutes=10),
    )
    torch.cuda.set_device(rank)

    saved = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    config = resolve_config(args, saved)
    exact = bool(saved and args.resume_mode == "exact")
    if exact:
        validate_exact_resume(saved, config, args, world_size)

    # Same model initialization; distinct process sampling streams are set below.
    torch.manual_seed(args.seed)
    environment = InstanceGenerator(config).generate(
        seed=args.seed + rank,
        randomize_uav=bool(config.get("ppo", {}).get("randomize_uav", True)),
    )
    if exact:
        environment = saved["rank_runtime"][rank]["environment"]

    actor = CoveragePolicy(environment, config=config).to(rank)
    agent = PPOAgent(actor, config, device=rank)
    start_update = 1
    if saved:
        last = agent.load_checkpoint(args.resume, restore_rng=False, load_optimizers=exact)
        start_update = last + 1 if exact else 1
        if not exact:
            agent.update_count = 0
        if rank == 0:
            print(f"[rank 0] {args.resume_mode}: {args.resume} -> update {start_update}", flush=True)

    initial_actor_sha256 = parameter_sha256(agent.actor)
    agent.actor = DDP(agent.actor, device_ids=[rank])
    agent.critic = DDP(agent.critic, device_ids=[rank])
    trainer = DistributedTrainer(environment, agent, rank=rank, world_size=world_size,
                                 checkpoint_dir=args.checkpoint_dir,
                                 checkpoint_every=args.checkpoint_every, config=config)
    trainer.run_metadata = {"seed": args.seed, "num_updates": args.num_updates,
                            "rollout_steps": args.rollout_steps, "world_size": world_size,
                            "initial_actor_sha256": initial_actor_sha256}
    if exact:
        runtime = saved["rank_runtime"][rank]
        trainer._count_emas = runtime["count_emas"]
        trainer._rank_stats = deepcopy(runtime.get("rank_stats", trainer._empty_rank_stats()))
        restore_rng_state(runtime["rng"])
    else:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
    if rank == 0 and args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"resolved_config": config, "training_metadata": trainer.run_metadata,
                    "resume": args.resume, "resume_mode": args.resume_mode,
                    "initial_actor_sha256": initial_actor_sha256}
        (directory / f"resolved_config_update_{start_update}.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Resolved PPO config: {json.dumps(config['ppo'])}", flush=True)
    trainer.train(args.num_updates, args.rollout_steps, start_update=start_update)
    assert_rank_parameters_equal(agent.actor, agent.critic)
    dist.barrier()
    dist.destroy_process_group()


def resolve_config(args, saved=None):
    """Defaults < checkpoint config < --set < explicitly supplied dedicated flags."""
    config = deepcopy(saved["resolved_config"]) if saved and saved.get("resolved_config") else load_config()
    if getattr(args, "set", None):
        config = _apply_overrides(config, args.set)
    # A fixed count pins the configured range; otherwise the environment draws
    # from [min, max] and resamples fields + UAV parameters on every reset.
    if args.num_fields is not None:
        config["environment"]["num_fields"] = {
            "min": args.num_fields, "max": args.num_fields,
        }
    # Only override what the user actually passed. Non-None defaults used to
    # overwrite ``--set ppo.epochs=...`` (requested 1, effective 4), silently
    # invalidating ablations.
    ppo_overrides = {}
    if args.ppo_epochs is not None:
        ppo_overrides["epochs"] = args.ppo_epochs
    if args.batch_size is not None:
        ppo_overrides["batch_size"] = args.batch_size
    if args.gae_lambda is not None:
        ppo_overrides["gae_lambda"] = args.gae_lambda
    config["ppo"].update(ppo_overrides)

    return config


def validate_exact_resume(saved, config, args, world_size):
    """Refuse to call a weights-only restart an exact continuation."""
    if not saved.get("rank_runtime") or not saved.get("resolved_config"):
        raise ValueError("Legacy checkpoint cannot resume exactly; use --resume-mode finetune.")
    if config != saved["resolved_config"]:
        raise ValueError("Exact resume cannot change configuration; use --resume-mode finetune.")
    meta = saved.get("training_metadata", {})
    for key, value in {"world_size": world_size, "num_updates": args.num_updates,
                       "rollout_steps": args.rollout_steps, "seed": args.seed}.items():
        if meta.get(key) != value:
            raise ValueError(f"Exact resume requires unchanged {key}; use --resume-mode finetune.")
    if len(saved["rank_runtime"]) != world_size:
        raise ValueError("Checkpoint is missing rank runtime states.")


def _apply_overrides(config: dict, overrides) -> dict:
    """Apply ``key.path=value`` overrides to a loaded config in place."""
    import json

    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = config
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return config


def main() -> None:
    """Parse arguments and launch one process per GPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-fields", type=int, default=None,
        help="Fixed field count for every episode; default samples the "
             "configured [min, max] range and resamples on each reset.",
    )
    parser.add_argument("--num-updates", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None,
                        help="Override ppo.epochs. Default None keeps the config value.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override ppo.batch_size. Default None keeps the config value.")
    parser.add_argument(
        "--gae-lambda", type=float, default=None,
        help="Override ppo.gae_lambda. Lower trades advantage bias for much "
             "lower variance, which only pays off with an accurate critic.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="coverage_env/results/checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE",
        help="Override one config value, e.g. "
             "--set rl.decoder.candidate_attention=false. Repeatable; values are "
             "parsed as JSON when possible (so true/false/null/0.0 work). This "
             "keeps experiments from having to edit default.yaml, which is how two "
             "runs previously ended up differing in ways nobody remembered.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint file to resume from; training continues at the next "
             "update index and still targets --num-updates overall.",
    )
    parser.add_argument("--resume-mode", choices=("exact", "finetune"), default="exact",
                        help="exact restores rank RNG/environments/config; finetune starts a new optimizer/schedule.")
    args = parser.parse_args()
    saved = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    metadata = saved.get("training_metadata", {}) if saved and args.resume_mode == "exact" else {}
    for key, default in {"num_updates": 200, "rollout_steps": 128, "seed": 42}.items():
        if getattr(args, key) is None:
            setattr(args, key, metadata.get(key, default))

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise SystemExit("No CUDA GPUs available.")
    if args.num_updates <= 0 or args.rollout_steps <= 0:
        raise SystemExit("num-updates and rollout-steps must be positive.")
    if args.checkpoint_every <= 0:
        raise SystemExit("checkpoint-every must be positive.")
    if saved and args.resume_mode == "exact":
        validate_exact_resume(saved, resolve_config(args, saved), args, world_size)

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
