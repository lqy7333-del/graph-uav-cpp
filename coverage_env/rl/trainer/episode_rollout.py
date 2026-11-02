"""Complete-episode PPO collection for fixed instance banks.

The legacy fixed-step mode remains in ``DistributedTrainer`` and is unchanged.
This module samples exactly ``episodes_per_rank`` complete episodes per rank,
never resets between updates just to fill a step budget, and pads only for DDP
synchronization (masked rows never enter a loss).
"""
from __future__ import annotations
import copy, json, math, time, traceback
from pathlib import Path
import torch
import torch.distributed as dist

from coverage_env.rl.ppo.loss import compute_advantage
from coverage_env.rl.buffer.rollout_buffer import RolloutBuffer
from coverage_env.rl.trainer.distributed import DistributedTrainer
from coverage_env.rl.trainer.overfit import OverfitTrainer, _write_json, aggregate_rows
from coverage_env.rl.evaluation.maneuver_eval import (
    run_episode, _nn_choice, _policy_choice_factory, replay_recorded,
)
from coverage_env.env.fixed_instance_pool import load_bank
from coverage_env.rl.checkpoint import capture_rng_state, restore_rng_state
from coverage_env.rl.trainer.episode import current_potential, learning_reward
from coverage_env.rl.trainer.trainer import _clone_graph


class EpisodeRolloutTrainer(OverfitTrainer):
    """Fixed-bank complete-episode sampler with masked uneven DDP updates."""
    def __init__(self, environment, agent, *, rank=0, world_size=1,
                 checkpoint_dir=None, checkpoint_every=10, config=None,
                 validation_bank=None, episodes_per_rank=4, max_episode_steps=500):
        super().__init__(environment, agent, rank=rank, world_size=world_size,
                         checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
                         config=config)
        self.episodes_per_rank = int(episodes_per_rank)
        self.max_episode_steps = int(max_episode_steps)
        self.validation_bank = validation_bank
        self._episode_open = False
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._episode_time = 0.0
        self._episode_area = 0.0
        self._episode_failed = 0
        self._last_state = None
        self._stats_completed = 0
        self._stats_truncated = 0
        self._stats_failed = 0
        self._stats_episodes = 0
        self._last_episode_lengths = []

    def _save_checkpoint(self, update_idx):
        # Save weights/runtime first, then evaluate train and optional validation
        # banks with RNG and optimizer state untouched.
        DistributedTrainer._save_checkpoint(self, update_idx)
        self.evaluate_bank(update_idx)
        if self.validation_bank:
            self.evaluate_validation(update_idx)

    def evaluate_validation(self, update_idx):
        rng = capture_rng_state()
        actor = self.agent.actor.module if hasattr(self.agent.actor, "module") else self.agent.actor
        was_training = actor.training
        local = {"rows": [], "nn": [], "error": None}
        try:
            actor.eval()
            manifest, templates = load_bank(self.validation_bank)
            choose = _policy_choice_factory(actor)
            for index in range(self.rank, len(templates), self.world_size):
                env = copy.deepcopy(templates[index])
                row = run_episode(env, choose, record=True)
                row.update(index=index, seed=manifest["instances"][index]["seed"])
                replay = replay_recorded(copy.deepcopy(templates[index]), row["actions"])
                if (not replay["completed"] or abs(replay["total_time"] - row["total_time"]) > 1e-6):
                    raise RuntimeError(f"recorded validation replay failed for instance {index}")
                row["replay_ok"] = True
                local["rows"].append(row)
                if update_idx == 0:
                    base = run_episode(copy.deepcopy(templates[index]), _nn_choice, record=True)
                    base.update(index=index, seed=row["seed"])
                    if not base["completed"]:
                        raise RuntimeError("NN cannot complete validation instance")
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
            raise RuntimeError(f"validation evaluation failed: {[g['error'] for g in gathered]}")
        if self.rank == 0:
            directory = Path(self.checkpoint_dir) / "validation"
            directory.mkdir(parents=True, exist_ok=True)
            rows = sorted([r for g in gathered for r in g["rows"]], key=lambda r: r["index"])
            expected = list(range(len(templates)))
            if [r["index"] for r in rows] != expected:
                raise RuntimeError("validation evaluation did not cover the bank")
            if update_idx == 0:
                refs = {"initial": rows, "nn": sorted(
                    [r for g in gathered for r in g["nn"]], key=lambda r: r["index"])}
                _write_json(directory / "references.json", refs)
            else:
                refs = json.loads((directory / "references.json").read_text(encoding="utf-8"))
            summary = aggregate_rows(rows, refs)
            summary.update(update=update_idx, scope="validation_only")
            _write_json(directory / "evaluation" / f"update_{update_idx}.json",
                        {"summary": summary, "rows": rows})
            best_path = directory / "best.json"
            best = json.loads(best_path.read_text(encoding="utf-8")) if best_path.exists() else None
            from coverage_env.rl.trainer.overfit import candidate_key
            if best is None or candidate_key(summary) < candidate_key(best):
                _write_json(best_path, summary)
            history = [json.loads(p.read_text(encoding="utf-8"))["summary"]
                       for p in (directory / "evaluation").glob("update_*.json")]
            history.sort(key=lambda r: r["update"])
            keys = sorted(set().union(*(r.keys() for r in history)))
            import csv
            with (directory / "validation_curve.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader(); writer.writerows(history)

    def _all_rank_phase_check(self, local_error, *, update_idx, phase):
        """Synchronize a sampling/gate-construction failure across all ranks.

        Every rank reaches this method even when its own phase raised. A single
        failing rank therefore cannot leave the others entering a collective
        alone. Rank 0 writes the gathered diagnostics before all ranks raise.
        """
        flag = torch.tensor([1.0 if local_error else 0.0], dtype=torch.float64,
                            device=self.agent.device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        any_error = bool(float(flag.item()) > 0.0)
        if not any_error:
            return
        gathered = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(gathered, local_error)
        else:
            gathered = [local_error]
        if self.rank == 0 and self.checkpoint_dir is not None:
            directory = Path(self.checkpoint_dir) / "run_failures"
            directory.mkdir(parents=True, exist_ok=True)
            payload = {"update": update_idx, "phase": phase,
                       "errors": gathered, "config": self.config,
                       "run_metadata": self.run_metadata}
            path = directory / f"update_{update_idx}_{phase}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                       default=str), encoding="utf-8")
        if dist.is_initialized():
            dist.barrier()
        raise RuntimeError(f"all-rank {phase} failure at update {update_idx}: {gathered}")

    def _sample_action(self, state):
        with torch.no_grad():
            output = self.agent.actor(state, environment=self.environment)
            distribution = output["action_distribution"]
            if distribution.numel() == 0 or float(distribution.detach().sum()) <= 0:
                raise RuntimeError("No feasible action from a live episode; collection must not reset to hide it.")
            index = int(torch.multinomial(distribution, 1).item())
            action = output["actions"][index]
            log_prob = torch.log(distribution[index].clamp_min(1e-12))
            value = self.agent.value_from_output(output)
        return output, action, log_prob, value

    def _start_episode(self):
        state = self.environment.reset()
        self._note_instance()
        self._record_nn_baseline()
        self._episode_open = True
        self._episode_steps = 0
        self._episode_reward = self._episode_time = self._episode_area = 0.0
        self._episode_failed = 0
        self._last_state = state
        return state

    def _close_episode(self, success, truncated, closure_error):
        self._stats_episodes += 1
        self._stats_completed += int(success)
        self._stats_truncated += int(truncated)
        self._stats_failed += int((not success) and (not truncated))
        self._rank_stats["episodes"] += 1
        count_here = len(self.environment.fields)
        self._rank_stats["field_counts"][str(count_here)] = (
            self._rank_stats["field_counts"].get(str(count_here), 0) + 1)
        self._last_episode_lengths.append(self._episode_steps)
        self._episode_open = False

    def _collect_episodes(self):
        self.buffer.clear()
        self.agent.actor.eval()
        actor_module = self.agent.actor.module if hasattr(self.agent.actor, "module") else self.agent.actor
        reward_mode = str(self.config.get("rl", {}).get("reward", {}).get("mode", "distance")).lower()
        reward_cfg = self.config.get("rl", {}).get("reward", {})
        width = float(reward_cfg.get("coverage_width", 10.0))
        nominal = float(reward_cfg.get("nominal_work_speed", 8.0))
        time_scale = float(reward_cfg.get("time_scale", 100.0))
        failure_penalty = float(self.config.get("rl", {}).get("failure_penalty", -1.0))
        closure_max = 0.0
        completed_episodes = 0
        truncated_episodes = 0
        if not self._episode_open:
            state = self._start_episode()
        else:
            state = self._last_state
        while completed_episodes < self.episodes_per_rank:
            episode_truncated = False
            episode_completed = False
            episode_success = False
            while True:
                if self._episode_steps >= self.max_episode_steps:
                    episode_truncated = True
                    break
                output, action, log_prob, value = self._sample_action(state)
                before = current_potential(self.environment, self.config)
                next_state, reward, env_done, info = self.environment.step(action)
                shaped = learning_reward(self.environment, reward, before, env_done,
                                         self.config, self.agent.gamma)
                self._episode_reward += float(reward)
                self._episode_time += float(info.get("time_total", 0.0) or 0.0)
                self._episode_area += float(info.get("coverage_gain", 0.0) or 0.0)
                self._episode_failed += int(bool(info.get("failed")))
                next_graph = actor_module.graph_builder.build_graph(self.environment)
                if env_done:
                    next_value = 0.0
                else:
                    _emb, _global, critic_input = actor_module.encode_graph(
                        next_graph, environment=self.environment, state=next_state)
                    next_value = self.agent.critic(critic_input).reshape(()).item()
                truncated = False
                self.buffer.add(
                    {"observation": copy.deepcopy(state),
                     "graph": _clone_graph(output["graph"]),
                     "actions": list(output["actions"]),
                     "mask": output["mask"].detach().cpu().tolist()},
                    action, shaped, {"observation": copy.deepcopy(next_state),
                                     "graph": _clone_graph(next_graph)},
                    env_done, log_prob.detach().item(), value.detach().item(),
                    next_value, truncated=truncated,
                )
                self._rank_stats["transitions"] += 1
                self._episode_steps += 1
                state = next_state
                if env_done:
                    episode_completed = True
                    episode_success = bool(info.get("success")
                                           and info.get("termination_reason") == "mission_complete")
                    break
            if episode_truncated and not episode_completed:
                # Mark the safety tail; it is not a successful complete episode.
                last = self.buffer.transitions[-1]
                last.truncated = True
                truncated_episodes += 1
            if reward_mode == "time":
                expected = -(self._episode_time - self._episode_area / (width * nominal)) / time_scale \
                           + failure_penalty * self._episode_failed
                closure_max = max(closure_max, abs(self._episode_reward - expected))
            self._close_episode(episode_success, episode_truncated, closure_max)
            completed_episodes += 1
            self._last_state = state
            if completed_episodes < self.episodes_per_rank:
                state = self._start_episode()
        # Do not reset after the fourth episode: that would create a zero-step
        # instance at the batch boundary.  The next update starts with reset().
        if truncated_episodes:
            raise RuntimeError(
                f"safety truncation in {truncated_episodes} episode(s); "
                "collection is not a complete-episode batch")
        if self.rank == 0:
            print(f"[episode] rank0 collected {completed_episodes} episodes, "
                  f"transitions={len(self.buffer.transitions)}, "
                  f"completed={self._stats_completed}, failed={self._stats_failed}, "
                  f"truncated={self._stats_truncated}", flush=True)
        return closure_max

    def train(self, num_updates: int, rollout_steps: int | None = None, *, start_update=1, stop_at=None):
        self.run_metadata.update(
            num_updates=num_updates, collection_mode="complete_episodes",
            episodes_per_rank=self.episodes_per_rank,
            max_episode_steps=self.max_episode_steps,
            world_size=self.world_size,
        )
        last_update = num_updates if stop_at is None else min(num_updates, stop_at)
        if self.checkpoint_dir and start_update == 1:
            self._save_checkpoint(0)
        for update_idx in range(max(1, start_update), last_update + 1):
            started = time.perf_counter()
            before_stats = (self._stats_completed, self._stats_failed, self._stats_truncated)
            collect_error = None
            try:
                closure = self._collect_episodes()
            except Exception as exc:
                collect_error = {"rank": self.rank, "phase": "collect",
                                 "error": repr(exc), "traceback": traceback.format_exc()}
            self._all_rank_phase_check(collect_error, update_idx=update_idx, phase="collect")
            transitions = list(self.buffer.transitions)
            adv, returns = compute_advantage(
                [t.reward for t in transitions], [t.value for t in transitions],
                [t.next_value for t in transitions], [t.done for t in transitions],
                self.agent.gamma, self.agent.gae_lambda, device=self.agent.device,
                truncated=[t.truncated for t in transitions],
            )
            local_count = len(transitions)
            stats_t = torch.tensor([float(local_count), float(adv.sum()), float((adv ** 2).sum())],
                                   dtype=torch.float64, device=self.agent.device)
            if dist.is_initialized():
                dist.all_reduce(stats_t, op=dist.ReduceOp.SUM)
            global_count = int(stats_t[0].item())
            mean = float(stats_t[1].item() / max(1, global_count))
            var = max(0.0, float(stats_t[2].item() / max(1, global_count) - mean * mean))
            adv = (adv - mean) / math.sqrt(var + 1e-8)
            if dist.is_initialized():
                max_len = torch.tensor([float(local_count)], device=self.agent.device)
                dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
                global_steps = int(max_len.item())
            else:
                global_steps = local_count
            # Construct gate metrics defensively. A rank-local exception here
            # must be synchronized before any rank enters _gate_or_abort's
            # collectives.
            gate_error = None
            try:
                local_gate = self._local_gate_metrics(transitions, closure) if self.value_gate_enabled else {
                    "value_state_gap_max": 0.0, "value_state_scale": 1.0, "pair_q_max": 0.0,
                    "tail_value_error": 0.0, "tail_value_scale": 1.0, "tail_q": 0.0,
                    "replay_logprob_error": 0.0, "replay_value_error": 0.0,
                    "episode_closure_error_max": closure,
                    "finite": all(math.isfinite(t.reward) and math.isfinite(t.value)
                                  and math.isfinite(t.next_value) for t in transitions),
                }
            except Exception as exc:
                local_gate = None
                gate_error = {"rank": self.rank, "phase": "gate_construct",
                              "error": repr(exc), "traceback": traceback.format_exc()}
            self._all_rank_phase_check(gate_error, update_idx=update_idx, phase="gate_construct")
            if self.value_gate_enabled:
                gate = self._gate_or_abort(local_gate, update_idx)
            else:
                gate = local_gate
            progress = self.agent.set_training_progress((update_idx - 1) / max(1, num_updates - 1))
            stats = self.agent.update_masked(
                transitions, [True] * local_count,
                global_valid_count=global_count, global_steps=global_steps,
                advantages=adv.detach().cpu(), returns=returns.detach().cpu())
            if self.value_gate_enabled:
                self._post_update_gate(stats, update_idx)
            self.buffer.clear()
            self._rank_stats["optimizer_steps"] += int(stats.get("updates", 0))
            self._rank_stats["updates"] += 1
            self._rank_stats["wall_seconds"] += time.perf_counter() - started
            if self.rank == 0:
                row = {
                    "update": update_idx, "episodes": self._stats_episodes,
                    "episodes_this_update": self.episodes_per_rank,
                    "transitions_this_update": local_count,
                    "transitions_total_rank0": self._rank_stats["transitions"],
                    "completed_this_update": self._stats_completed - before_stats[0],
                    "failed_this_update": self._stats_failed - before_stats[1],
                    "truncated_this_update": self._stats_truncated - before_stats[2],
                    "optimizer_steps_total": self._rank_stats["optimizer_steps"],
                    "update_wall_seconds": round(float(self._rank_stats["wall_seconds"]), 3),
                    "entropy_coef": round(float(progress["entropy_coef"]), 6),
                    "actor_learning_rate": round(float(progress["actor_learning_rate"]), 8),
                    "actor_loss": round(stats["actor_loss"], 6),
                    "critic_loss": round(stats["critic_loss"], 6),
                    "entropy": round(stats["entropy"], 5),
                    "policy_loss": round(stats["policy_loss"], 6),
                    "approx_kl": round(stats["approx_kl"], 7),
                    "clip_fraction": round(stats["clip_fraction"], 5),
                    "ratio_std": round(stats["ratio_std"], 6),
                    "advantage_std": round(stats["advantage_std"], 5),
                    "return_mean": round(stats["return_mean"], 5),
                    "return_std": round(stats["return_std"], 5),
                    "value_mean": round(stats["value_mean"], 5),
                    "value_std": round(stats["value_std"], 5),
                    "explained_variance": round(stats["explained_variance"], 6),
                    "actor_grad_norm": round(stats["actor_grad_norm"], 5),
                    "value_gate_passed": True,
                    "pair_q_max": round(gate["pair_q_max"], 8),
                    "tail_q": round(gate["tail_q"], 8),
                    "replay_logprob_error": round(gate["replay_logprob_error"], 8),
                    "replay_value_error": round(gate["replay_value_error"], 8),
                    "episode_closure_error_max": round(gate["episode_closure_error_max"], 12),
                    "global_valid_transitions": global_count,
                    "global_steps": global_steps,
                }
                self._append_log(row)
                print(f"[episode] update {update_idx} episodes={self.episodes_per_rank} "
                      f"transitions={local_count} global={global_count} "
                      f"actor_loss={stats['actor_loss']:.5f} critic_loss={stats['critic_loss']:.5f} "
                      f"updates={stats['updates']}", flush=True)
            if self.checkpoint_dir and (update_idx % self.checkpoint_every == 0 or update_idx == num_updates):
                self._save_checkpoint(update_idx)
        self.write_run_summary()
        if self.rank == 0 and self.checkpoint_dir is not None:
            path = Path(self.checkpoint_dir) / "run_summary.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["bank_usage"] = self.environment.usage()
                path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                encoding="utf-8")
