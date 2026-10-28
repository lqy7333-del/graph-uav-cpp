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
import os
import socket
import json
import random
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
from coverage_env.rl.checkpoint import capture_rng_state, restore_rng_state, assert_rank_parameters_equal


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
            # Sampling is stochastic and must not feed the autograd graph.
            self.agent.actor.eval()
            state = self.environment.reset()
            self._record_nn_baseline()
            rollout_reward = 0.0
            # Under field-count generalization a 128-step rollout spans several
            # episodes with different field counts, so the rollout sum alone is
            # dominated by instance variance. Accumulate per-episode returns and
            # label each with the episode's own field count so the log can show
            # a trend and compare magnitudes across field counts.
            episode_reward = 0.0
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
                if env_done:
                    # The environment still describes the episode that just
                    # finished, so its field count labels this episode.
                    completed_rewards.append(episode_reward)
                    count_here = len(self.environment.fields)
                    completed_fields.append(count_here)
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
                                next_graph, environment=self.environment
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
                state = self.environment.reset() if env_done else next_state
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
            # Advance entropy annealing before the update so the logged value is
            # the one actually used.
            progress = self.agent.set_training_progress(
                (update_idx - 1) / max(1, num_updates - 1)
            )
            current_entropy_coef = progress["entropy_coef"]
            stats = self.agent.update(self.buffer)
            self.buffer.clear()

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
                 "count_emas": deepcopy(self._count_emas)}
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
    dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)
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

    agent.actor = DDP(agent.actor, device_ids=[rank])
    agent.critic = DDP(agent.critic, device_ids=[rank])
    trainer = DistributedTrainer(environment, agent, rank=rank, world_size=world_size,
                                 checkpoint_dir=args.checkpoint_dir,
                                 checkpoint_every=args.checkpoint_every, config=config)
    trainer.run_metadata = {"seed": args.seed, "num_updates": args.num_updates,
                            "rollout_steps": args.rollout_steps, "world_size": world_size}
    if exact:
        runtime = saved["rank_runtime"][rank]
        trainer._count_emas = runtime["count_emas"]
        restore_rng_state(runtime["rng"])
    else:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
    if rank == 0 and args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"resolved_config": config, "training_metadata": trainer.run_metadata,
                    "resume": args.resume, "resume_mode": args.resume_mode}
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
