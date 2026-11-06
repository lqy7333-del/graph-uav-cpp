"""CLI for random-instance, fixed-10-field complete-episode PPO."""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, random, shutil
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from coverage_env.env.fixed_instance_pool import load_bank, file_sha256
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.distributed import _free_port
from coverage_env.rl.trainer.episode_random import RandomInstanceEpisodeTrainer
from coverage_env.rl.trainer.overfit import _LocalModule, candidate_key
from coverage_env.rl.checkpoint import parameter_sha256, restore_rng_state


# None distinguishes omitted options from explicit resume overrides.
_RUN_DEFAULTS = dict(num_updates=1200, episodes_per_rank=4, max_episode_steps=500,
                     seed=42, instance_seed_base=900_000_000, num_fields=10,
                     actor_lr=1e-4, critic_lr=3e-3, eval_every=20)


def resolve_run_arguments(args, saved=None):
    for name, default in _RUN_DEFAULTS.items():
        if saved is None:
            expected = default
        elif name in ("actor_lr", "critic_lr"):
            key = "actor_learning_rate" if name == "actor_lr" else "critic_learning_rate"
            expected = saved["resolved_config"]["ppo"][key]
        else:
            expected = saved["training_metadata"][name]
        requested = getattr(args, name)
        if saved is not None and requested is not None and requested != expected:
            raise ValueError(f"resume {name} changed: {requested!r} != {expected!r}")
        setattr(args, name, expected if requested is None else requested)
    if args.num_fields != 10:
        raise ValueError("this protocol requires exactly 10 fields")
    if saved is not None:
        if saved['training_metadata'].get('collection_mode') != 'complete_episodes':
            raise ValueError('collection mode changed')


def validate_optimizer_lrs(agent, config):
    # The frozen protocol uses no LR schedule. Check loaded optimizer values,
    # not merely the config/logging attributes.
    if config['ppo'].get('lr_schedule', 'none') != 'none':
        raise ValueError('random-instance protocol requires lr_schedule=none')
    for kind in ('actor', 'critic'):
        expected = config['ppo'][kind + '_learning_rate']
        optimizer = getattr(agent, kind + '_optimizer')
        if any(group['lr'] != expected for group in optimizer.param_groups):
            raise ValueError(f'{kind} optimizer LR differs from resolved configuration')


def restore_panel_history(source, out, cutoff):
    """Rebuild selection from history, preserving the selected weights.

    Future evaluations relative to the resumed checkpoint are never imported.
    Same-directory rewind is refused instead of silently mixing two histories.
    """
    source, out = Path(source).resolve(), Path(out).resolve()
    for panel in ('diagnostic', 'validation'):
        src, dst = source / panel, out / panel
        records = []
        for path in (src / 'evaluation').glob('update_*.json'):
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['summary']['update'] > cutoff:
                if source == out:
                    raise ValueError('same-directory resume has evaluations newer than checkpoint; use a new output')
                continue
            records.append((path, record))
        if not records:
            raise ValueError(f'missing {panel} evaluation history for resume')
        if source != out:
            if (dst / 'evaluation').exists():
                raise ValueError('resume output already contains evaluation history')
            (dst / 'evaluation').mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / 'references.json', dst / 'references.json')
            for path, _ in records:
                shutil.copy2(path, dst / 'evaluation' / path.name)
        history = sorted((r['summary'] for _, r in records), key=lambda r: r['update'])
        best = min(history, key=candidate_key)
        weights = source / f"ckpt_update_{best['update']}.pt"
        if not weights.is_file():
            raise ValueError(f'missing historical best checkpoint: {weights}')
        if source != out:
            shutil.copy2(weights, out / weights.name)
        (dst / 'best.json').write_text(json.dumps(best, indent=2), encoding='utf-8')
        import csv
        keys = sorted(set().union(*(r.keys() for r in history)))
        with (dst / f'{panel}_curve.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(history)


def _worker(rank, world, args):
    distributed=args.device=='cuda'
    if distributed:
        torch.cuda.set_device(rank)
        dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=world,
                                timeout=datetime.timedelta(minutes=20))
    try:
        torch.set_num_threads(1)
        device=torch.device(f'cuda:{rank}' if distributed else 'cpu')
        saved=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
        resolve_run_arguments(args, saved)
        if saved is not None:
            config=saved['resolved_config']; meta=saved['training_metadata']
            if meta.get('instance_mode')!='random_instances': raise ValueError('not random-instance checkpoint')
            if int(meta.get('num_fields',-1))!=int(args.num_fields): raise ValueError('num_fields changed')
            if int(meta.get('num_updates',-1))!=int(args.num_updates): raise ValueError('num_updates changed')
            if int(meta.get('seed',-1))!=int(args.seed): raise ValueError('seed changed')
            if int(meta.get('world_size',-1))!=world: raise ValueError('world_size changed')
            if int(meta.get('episodes_per_rank',-1))!=int(args.episodes_per_rank): raise ValueError('episodes/rank changed')
            if int(meta.get('max_episode_steps',-1))!=int(args.max_episode_steps): raise ValueError('max steps changed')
            if int(meta.get('eval_every',-1))!=int(args.eval_every): raise ValueError('eval_every changed')
            if int(meta.get('instance_seed_base',-1))!=int(args.instance_seed_base): raise ValueError('instance seed base changed')
            if meta.get('diagnostic_bank_sha')!=file_sha256(Path(args.diagnostic_bank)/'manifest.json'): raise ValueError('diagnostic bank changed')
            if meta.get('validation_bank_sha')!=file_sha256(Path(args.validation_bank)/'manifest.json'): raise ValueError('validation bank changed')
            if meta.get('generator_config_sha') != hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest():
                raise ValueError('generator configuration changed')
            environment=saved['rank_runtime'][rank]['environment']
            stream_state=saved['rank_runtime'][rank].get('instance_stream')
            if not stream_state: raise ValueError('checkpoint lacks instance stream')
            start_update=int(saved.get('update_count',0))+1
        else:
            config=json.loads((Path(args.diagnostic_bank)/'manifest.json').read_text())['config']
            environment=load_bank(args.diagnostic_bank)[1][0]  # policy template only
            stream_state=None; start_update=1
        variant = getattr(args, 'encoder_variant', None)
        reference = getattr(args, 'paired_init', None)
        paired_metadata = None
        if saved is not None:
            actual = config['rl']['encoder'].get('variant', 'chanet')
            if variant is not None and variant != actual:
                raise ValueError('resume encoder variant changed')
            if reference is not None:
                raise ValueError('resume uses saved weights; do not pass paired-init')
            paired_metadata = saved['training_metadata'].get('paired_initialization')
        else:
            variant = variant or 'chanet'
            if variant == 'node_mlp' and reference is None:
                raise ValueError('node_mlp experiment requires --paired-init A ckpt_update_0.pt')
            if variant == 'chanet' and reference is not None:
                raise ValueError('paired-init is reserved for the node_mlp ablation')
        config=json.loads(json.dumps(config))
        if saved is None:
            config['environment']['num_fields']={'min':10,'max':10}
            config['ppo'].update(collection_mode='complete_episodes', episodes_per_rank=args.episodes_per_rank,
                                 max_episode_steps=args.max_episode_steps, randomize_uav=False,
                                 value_gate=True, actor_learning_rate=args.actor_lr,
                                 critic_learning_rate=args.critic_lr)
        if saved is None and variant == 'node_mlp':
            config['rl']['encoder']['variant'] = variant
        torch.manual_seed(args.seed)
        agent=PPOAgent(CoveragePolicy(environment,config=config),config,device=device)
        if saved is not None:
            agent.load_checkpoint(args.resume,map_location=device,restore_rng=False,load_optimizers=True)
            restore_rng_state(saved['rank_runtime'][rank]['rng'])
        if saved is None and reference is not None:
            from coverage_env.rl.trainer.chanet_ablation import initialize_from_a_zero
            paired_metadata = initialize_from_a_zero(agent, reference, config, {
                'seed': args.seed, 'num_updates': args.num_updates, 'world_size': world,
                'episodes_per_rank': args.episodes_per_rank, 'num_fields': 10,
                'instance_mode': 'random_instances', 'collection_mode': 'complete_episodes',
                'instance_seed_base': args.instance_seed_base, 'max_episode_steps': args.max_episode_steps,
                'eval_every': args.eval_every,
                'diagnostic_bank_sha': file_sha256(Path(args.diagnostic_bank)/'manifest.json'),
                'validation_bank_sha': file_sha256(Path(args.validation_bank)/'manifest.json')})
        validate_optimizer_lrs(agent, config)
        initial_hash=(saved['training_metadata']['initial_actor_sha256'] if saved is not None
                      else parameter_sha256(agent.actor))
        agent.actor=DDP(agent.actor,device_ids=[rank]) if distributed else _LocalModule(agent.actor)
        agent.critic=DDP(agent.critic,device_ids=[rank]) if distributed else agent.critic
        trainer=RandomInstanceEpisodeTrainer(
            environment,agent,rank=rank,world_size=world,checkpoint_dir=args.output,
            checkpoint_every=args.checkpoint_every,config=config,
            diagnostic_bank=args.diagnostic_bank,validation_bank=args.validation_bank,
            instance_seed_base=args.instance_seed_base,num_fields=10,
            episodes_per_rank=args.episodes_per_rank,max_episode_steps=args.max_episode_steps,
            eval_every=args.eval_every)
        if stream_state:
            trainer.stream.restore(stream_state)
            trainer._episode_seq=int(saved['rank_runtime'][rank].get('episode_seq',stream_state['counter']))
            trainer._episode_open=False
            trainer._rank_stats=saved['rank_runtime'][rank].get('rank_stats',trainer._empty_rank_stats()).copy()
        trainer.run_metadata={
            'seed':args.seed,'num_updates':args.num_updates,'world_size':world,
            'collection_mode':'complete_episodes','instance_mode':'random_instances',
            'num_fields':10,'episodes_per_rank':args.episodes_per_rank,
            'max_episode_steps':args.max_episode_steps,'instance_seed_base':args.instance_seed_base,
            'eval_every':args.eval_every,
            'generator_config_sha':trainer.generator_config_sha,
            'diagnostic_bank_sha':file_sha256(Path(args.diagnostic_bank)/'manifest.json'),
            'validation_bank_sha':file_sha256(Path(args.validation_bank)/'manifest.json'),
            'initial_actor_sha256':initial_hash,'initialization':None,
            'purpose':'random_instance_fixed_10field_generalization',
            'encoder_variant': config['rl']['encoder'].get('variant', 'chanet'),
            'paired_initialization': paired_metadata}
        if saved is None:
            random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
        preparation_error = None
        try:
            if rank==0:
                Path(args.output).mkdir(parents=True,exist_ok=True)
                if saved is not None:
                    source=Path(args.resume).resolve().parent; out=Path(args.output).resolve()
                    restore_panel_history(source, out, start_update - 1)
                (Path(args.output)/f'resolved_config_update_{start_update}.json').write_text(
                    json.dumps({'resolved_config':config,'training_metadata':trainer.run_metadata},indent=2),encoding='utf-8')
        except Exception as exc:
            preparation_error = {"rank": rank, "error": repr(exc)}
        trainer._all_rank_phase_check(preparation_error, update_idx=start_update,
                                      phase="resume_history")
        stop_options = {} if getattr(args, 'stop_at', None) is None else {'stop_at': args.stop_at}
        trainer.train(args.num_updates,None,start_update=start_update, **stop_options)
        if distributed: dist.barrier()
    finally:
        if distributed and dist.is_initialized(): dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--diagnostic-bank',required=True); p.add_argument('--validation-bank',required=True)
    p.add_argument('--output',required=True); p.add_argument('--resume',default=None)
    p.add_argument('--num-updates',type=int,default=None); p.add_argument('--episodes-per-rank',type=int,default=None)
    p.add_argument('--max-episode-steps',type=int,default=None); p.add_argument('--seed',type=int,default=None)
    p.add_argument('--instance-seed-base',type=int,default=None); p.add_argument('--num-fields',type=int,default=None)
    p.add_argument('--actor-lr',type=float,default=None); p.add_argument('--critic-lr',type=float,default=None)
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--encoder-variant', choices=('chanet', 'node_mlp'), default=None)
    p.add_argument('--paired-init', default=None, help='A update-0 checkpoint for shared random weights only')
    p.add_argument('--stop-at', type=int, default=None, help='bounded smoke; schedules keep num-updates horizon')
    args=p.parse_args()
    ck = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    resolve_run_arguments(args, ck)
    world=torch.cuda.device_count() if args.device=='cuda' else 1
    if args.device=='cuda' and world<1: raise SystemExit('No CUDA GPUs')
    os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']=str(_free_port())
    if world==1 and args.device=='cpu': _worker(0,1,args)
    else: mp.spawn(_worker,args=(world,args),nprocs=world,join=True)

if __name__=='__main__': main()
