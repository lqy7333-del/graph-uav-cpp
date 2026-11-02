"""Worker/CLI for fixed-bank complete-episode PPO. No legacy mode is changed."""
from __future__ import annotations
import argparse, datetime, json, os, random, shutil
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from coverage_env.config import load_config
from coverage_env.env.fixed_instance_pool import FixedInstancePool
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.distributed import _free_port
from coverage_env.rl.trainer.episode_rollout import EpisodeRolloutTrainer
from coverage_env.rl.trainer.overfit import _LocalModule
from coverage_env.rl.checkpoint import parameter_sha256, restore_rng_state


def _worker(rank, world_size, args):
    distributed = args.device == "cuda"
    saved = torch.load(args.resume, map_location="cpu", weights_only=False) if getattr(args, "resume", None) else None
    if distributed:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size,
                                timeout=datetime.timedelta(minutes=20))
    try:
        torch.set_num_threads(1)
        device = torch.device(f"cuda:{rank}" if distributed else "cpu")
        if saved is not None:
            config = saved["resolved_config"]
            environment = saved["rank_runtime"][rank]["environment"]
            manifest = environment.manifest
            meta = saved.get("training_metadata", {})
            if meta.get("world_size") != world_size:
                raise ValueError("resume world_size does not match checkpoint")
            if int(meta.get("episodes_per_rank", -1)) != int(args.episodes_per_rank):
                raise ValueError("resume episodes_per_rank does not match checkpoint")
            if int(meta.get("num_updates", -1)) != int(args.num_updates):
                raise ValueError("resume num_updates does not match checkpoint")
            if int(meta.get("seed", -1)) != int(args.seed):
                raise ValueError("resume seed does not match checkpoint")
            if meta.get("collection_mode") != "complete_episodes":
                raise ValueError("resume checkpoint is not complete_episodes mode")
            from coverage_env.env.fixed_instance_pool import file_sha256
            args_bank_sha = file_sha256(Path(args.bank) / "manifest.json")
            if meta.get("bank_manifest_sha") != args_bank_sha:
                raise ValueError("resume --bank identity does not match checkpoint")
            if meta.get("bank_manifest_sha") != environment.manifest_sha:
                raise ValueError("resume training bank changed")
            if args.validation_bank:
                from coverage_env.env.fixed_instance_pool import file_sha256
                val_sha = file_sha256(Path(args.validation_bank) / "manifest.json")
                if meta.get("validation_bank_manifest_sha") != val_sha:
                    raise ValueError("resume validation bank changed")
        else:
            environment = FixedInstancePool(args.bank, seed=args.seed + rank)
            manifest = environment.manifest
            config = manifest["config"]
        config["ppo"]["collection_mode"] = "complete_episodes"
        config["ppo"]["episodes_per_rank"] = args.episodes_per_rank
        config["ppo"]["max_episode_steps"] = args.max_episode_steps
        # Fresh short fine-tune: weights only, new optimizers, new schedules.
        if args.initial_checkpoint and saved is None:
            ck = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
            config = ck["resolved_config"]
            config["environment"]["num_fields"] = {"min": 10, "max": 10}
            config["ppo"]["collection_mode"] = "complete_episodes"
            config["ppo"]["episodes_per_rank"] = args.episodes_per_rank
            config["ppo"]["max_episode_steps"] = args.max_episode_steps
            config["ppo"]["randomize_uav"] = False
            config["ppo"]["value_gate"] = True
        if saved is None:
            config["ppo"]["actor_learning_rate"] = float(args.actor_lr)
            config["ppo"]["critic_learning_rate"] = float(args.critic_lr)
            config["ppo"]["collection_mode"] = "complete_episodes"
            config["ppo"]["episodes_per_rank"] = args.episodes_per_rank
            config["ppo"]["max_episode_steps"] = args.max_episode_steps
            config["ppo"]["randomize_uav"] = False
            config["ppo"]["value_gate"] = True
        torch.manual_seed(args.seed)
        agent = PPOAgent(CoveragePolicy(environment, config=config), config, device=device)
        if saved is not None:
            agent.load_checkpoint(args.resume, map_location=device,
                                  restore_rng=False, load_optimizers=True)
            runtime = saved["rank_runtime"][rank]
            from coverage_env.rl.checkpoint import restore_rng_state
            restore_rng_state(runtime["rng"])
            start_update = int(saved.get("update_count", 0)) + 1
        elif args.initial_checkpoint:
            agent.load_checkpoint(args.initial_checkpoint, map_location=device,
                                  restore_rng=False, load_optimizers=False)
            agent.update_count = 0
            start_update = 1
            agent.actor_lr_start = float(args.actor_lr)
            agent.critic_lr_start = float(args.critic_lr)
            for group in agent.actor_optimizer.param_groups:
                group["lr"] = float(args.actor_lr)
            for group in agent.critic_optimizer.param_groups:
                group["lr"] = float(args.critic_lr)
        initial_hash = parameter_sha256(agent.actor)
        start_update = int(saved.get("update_count", 0)) + 1 if saved is not None else 1
        agent.actor = DDP(agent.actor, device_ids=[rank]) if distributed else _LocalModule(agent.actor)
        agent.critic = DDP(agent.critic, device_ids=[rank]) if distributed else agent.critic
        trainer = EpisodeRolloutTrainer(
            environment, agent, rank=rank, world_size=world_size,
            checkpoint_dir=args.output, checkpoint_every=args.checkpoint_every,
            config=config, episodes_per_rank=args.episodes_per_rank,
            max_episode_steps=args.max_episode_steps,
            validation_bank=getattr(args, "validation_bank", None))
        trainer.run_metadata = {
            "seed": args.seed, "num_updates": args.num_updates, "world_size": world_size,
            "collection_mode": "complete_episodes", "episodes_per_rank": args.episodes_per_rank,
            "max_episode_steps": args.max_episode_steps,
            "bank_manifest_sha": environment.manifest_sha,
            "initial_actor_sha256": initial_hash,
            "initialization": ({"checkpoint": str(args.initial_checkpoint),
                                "sha256": __import__("hashlib").sha256(Path(args.initial_checkpoint).read_bytes()).hexdigest()}
                               if args.initial_checkpoint else None),
            "purpose": "fixed_10field_complete_episode_finetune_or_random",
            "validation_bank_manifest_sha": (__import__("coverage_env.env.fixed_instance_pool", fromlist=["file_sha256"]).file_sha256(Path(args.validation_bank) / "manifest.json")
                                             if args.validation_bank else None),
        }
        if saved is None:
            random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
        if rank == 0:
            Path(args.output).mkdir(parents=True, exist_ok=True)
            if saved is not None:
                source = Path(args.resume).resolve().parent
                output_dir = Path(args.output).resolve()
                if source != output_dir:
                    for name, target in (("references.json", "references.json"),
                                         ("validation/references.json", "validation/references.json")):
                        if (source / name).exists():
                            (output_dir / target).parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source / name, output_dir / target)
            (Path(args.output) / "resolved_config_update_1.json").write_text(
                json.dumps({"resolved_config": config, "training_metadata": trainer.run_metadata},
                           indent=2, ensure_ascii=False), encoding="utf-8")
        if saved is not None:
            trainer._rank_stats = saved["rank_runtime"][rank].get("rank_stats", trainer._empty_rank_stats()).copy()
            trainer._count_emas = saved["rank_runtime"][rank].get("count_emas", {})
            trainer._episode_open = False
            # Resuming at a checkpoint boundary always starts the next update
            # with reset(); no mid-episode state is claimed.
        trainer.train(args.num_updates, None, start_update=start_update)
        if distributed:
            dist.barrier()
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bank", required=True); p.add_argument("--output", required=True)
    p.add_argument("--validation-bank", default=None)
    p.add_argument("--initial-checkpoint", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--num-updates", type=int, default=2)
    p.add_argument("--episodes-per-rank", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-3)
    p.add_argument("--device", choices=("cuda","cpu"), default="cuda")
    args=p.parse_args()
    if args.episodes_per_rank < 1: raise SystemExit("episodes-per-rank must be positive")
    # No initial checkpoint means fresh random actor/critic (the formal mode).
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        args.num_updates = int(ck["training_metadata"].get("num_updates", args.num_updates))
        args.seed = int(ck["training_metadata"].get("seed", args.seed))
    world_size=torch.cuda.device_count() if args.device=="cuda" else 1
    if args.device=="cuda" and world_size<1: raise SystemExit("No CUDA GPUs")
    os.environ["MASTER_ADDR"]="127.0.0.1"; os.environ["MASTER_PORT"]=str(_free_port())
    if world_size==1 and args.device=="cpu":
        _worker(0,1,args)
    else:
        mp.spawn(_worker,args=(world_size,args),nprocs=world_size,join=True)

if __name__=="__main__": main()
