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
from coverage_env.rl.trainer.overfit import _LocalModule
from coverage_env.rl.checkpoint import parameter_sha256, restore_rng_state


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
        if saved is not None:
            config=saved['resolved_config']; meta=saved['training_metadata']
            if meta.get('instance_mode')!='random_instances': raise ValueError('not random-instance checkpoint')
            if int(meta.get('num_fields',-1))!=int(args.num_fields): raise ValueError('num_fields changed')
            if int(meta.get('num_updates',-1))!=int(args.num_updates): raise ValueError('num_updates changed')
            if int(meta.get('seed',-1))!=int(args.seed): raise ValueError('seed changed')
            if int(meta.get('world_size',-1))!=world: raise ValueError('world_size changed')
            if int(meta.get('episodes_per_rank',-1))!=int(args.episodes_per_rank): raise ValueError('episodes/rank changed')
            if int(meta.get('max_episode_steps',-1))!=int(args.max_episode_steps): raise ValueError('max steps changed')
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
        config=json.loads(json.dumps(config))
        config['environment']['num_fields']={'min':10,'max':10}
        config['ppo'].update(collection_mode='complete_episodes', episodes_per_rank=args.episodes_per_rank,
                             max_episode_steps=args.max_episode_steps, randomize_uav=False,
                             value_gate=True, actor_learning_rate=args.actor_lr,
                             critic_learning_rate=args.critic_lr)
        torch.manual_seed(args.seed)
        agent=PPOAgent(CoveragePolicy(environment,config=config),config,device=device)
        if saved is not None:
            agent.load_checkpoint(args.resume,map_location=device,restore_rng=False,load_optimizers=True)
            restore_rng_state(saved['rank_runtime'][rank]['rng'])
        initial_hash=parameter_sha256(agent.actor)
        agent.actor=DDP(agent.actor,device_ids=[rank]) if distributed else _LocalModule(agent.actor)
        agent.critic=DDP(agent.critic,device_ids=[rank]) if distributed else agent.critic
        trainer=RandomInstanceEpisodeTrainer(
            environment,agent,rank=rank,world_size=world,checkpoint_dir=args.output,
            checkpoint_every=args.checkpoint_every,config=config,
            diagnostic_bank=args.diagnostic_bank,validation_bank=args.validation_bank,
            instance_seed_base=args.instance_seed_base,num_fields=10,
            episodes_per_rank=args.episodes_per_rank,max_episode_steps=args.max_episode_steps)
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
            'generator_config_sha':trainer.generator_config_sha,
            'diagnostic_bank_sha':file_sha256(Path(args.diagnostic_bank)/'manifest.json'),
            'validation_bank_sha':file_sha256(Path(args.validation_bank)/'manifest.json'),
            'initial_actor_sha256':initial_hash,'initialization':None,
            'purpose':'random_instance_fixed_10field_generalization'}
        if saved is None:
            random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
        if rank==0:
            Path(args.output).mkdir(parents=True,exist_ok=True)
            if saved is not None:
                source=Path(args.resume).resolve().parent; out=Path(args.output).resolve()
                if source!=out:
                    for name,target in (("references.json","references.json"),
                                        ("validation/references.json","validation/references.json"),
                                        ("diagnostic/references.json","diagnostic/references.json"),
                                        ("learning_curve.csv","learning_curve.csv"),
                                        ("validation/validation_curve.csv","validation/validation_curve.csv"),
                                        ("diagnostic/diagnostic_curve.csv","diagnostic/diagnostic_curve.csv")):
                        if (source/name).exists():
                            (out/target).parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source/name,out/target)
                    for dname in ('evaluation','validation/evaluation','diagnostic/evaluation'):
                        if (source/dname).is_dir(): shutil.copytree(source/dname,out/dname,dirs_exist_ok=True)
            (Path(args.output)/'resolved_config_update_1.json').write_text(
                json.dumps({'resolved_config':config,'training_metadata':trainer.run_metadata},indent=2),encoding='utf-8')
        trainer.train(args.num_updates,None,start_update=start_update)
        if distributed: dist.barrier()
    finally:
        if distributed and dist.is_initialized(): dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--diagnostic-bank',required=True); p.add_argument('--validation-bank',required=True)
    p.add_argument('--output',required=True); p.add_argument('--resume',default=None)
    p.add_argument('--num-updates',type=int,default=600); p.add_argument('--episodes-per-rank',type=int,default=4)
    p.add_argument('--max-episode-steps',type=int,default=500); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--instance-seed-base',type=int,default=900_000_000); p.add_argument('--num-fields',type=int,default=10)
    p.add_argument('--actor-lr',type=float,default=1e-4); p.add_argument('--critic-lr',type=float,default=3e-3)
    p.add_argument('--checkpoint-every',type=int,default=10); p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=p.parse_args()
    if args.num_fields!=10: raise SystemExit('this protocol requires exactly 10 fields')
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)
        args.num_updates=int(ck['training_metadata']['num_updates']); args.seed=int(ck['training_metadata']['seed'])
        args.instance_seed_base=int(ck['training_metadata']['instance_seed_base'])
    world=torch.cuda.device_count() if args.device=='cuda' else 1
    if args.device=='cuda' and world<1: raise SystemExit('No CUDA GPUs')
    os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']=str(_free_port())
    if world==1 and args.device=='cpu': _worker(0,1,args)
    else: mp.spawn(_worker,args=(world,args),nprocs=world,join=True)

if __name__=='__main__': main()
