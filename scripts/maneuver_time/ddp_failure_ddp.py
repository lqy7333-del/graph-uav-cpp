"""Real-DDP two-rank fault injection for complete-episode PPO.

Unlike the earlier helper, actor and critic are wrapped in
torch.nn.parallel.DistributedDataParallel. Sampling failure is injected on
rank 1 only after the first policy-forward sampling step has completed.
"""
import argparse, datetime, json, socket, sys, traceback
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.env.fixed_instance_pool import FixedInstancePool
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.episode_rollout import EpisodeRolloutTrainer


def worker(rank, world, port, bank, base, phase, backend):
    if backend=='nccl':
        torch.cuda.set_device(rank)
        device=torch.device(f'cuda:{rank}')
    else:
        device=torch.device('cpu')
    dist.init_process_group(backend, init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=180))
    try:
        config=json.loads((Path(bank)/'manifest.json').read_text())['config']
        config['ppo'].update(collection_mode='complete_episodes', episodes_per_rank=1,
                             batch_size=2, epochs=1, actor_learning_rate=1e-4,
                             critic_learning_rate=3e-3, value_gate=True,
                             randomize_uav=False)
        torch.manual_seed(42)
        pool=FixedInstancePool(bank, seed=42+rank)
        agent=PPOAgent(CoveragePolicy(pool, config=config), config, device=device)
        agent.actor=DDP(agent.actor, device_ids=[rank] if backend=='nccl' else None)
        agent.critic=DDP(agent.critic, device_ids=[rank] if backend=='nccl' else None)
        out=Path(base)/phase
        trainer=EpisodeRolloutTrainer(pool, agent, rank=rank, world_size=world,
                                      checkpoint_dir=out, checkpoint_every=1,
                                      config=config, episodes_per_rank=1,
                                      max_episode_steps=80)
        original_sample=trainer._sample_action
        state={'calls':0}
        def injected_sample(s):
            state['calls']+=1
            if rank==1 and phase=='collect' and state['calls']>=2:
                raise RuntimeError('injected sampling failure after first DDP forward')
            return original_sample(s)
        trainer._sample_action=injected_sample
        if rank==1 and phase=='gate':
            trainer._local_gate_metrics=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError('injected gate-construction failure after DDP sampling'))
        result={'rank':rank,'backend':backend,'phase':phase,'raised':False,'error':'',
                'update_count':None,'sample_calls':None,'log_exists':None,'diag_exists':None}
        try:
            trainer.train(1, rollout_steps=1)
        except Exception as exc:
            result['raised']=True; result['error']=repr(exc)
        result['update_count']=int(getattr(agent,'update_count',-1))
        result['sample_calls']=state['calls']
        result['log_exists']=(out/'train_log.csv').exists()
        result['diag_exists']=(out/'run_failures'/f'update_1_{"gate_construct" if phase=="gate" else "collect"}.json').exists()
        Path(base).mkdir(parents=True,exist_ok=True)
        (Path(base)/f'{phase}_rank{rank}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True); ap.add_argument('--phase',choices=('collect','gate'),required=True)
    ap.add_argument('--backend',choices=('gloo','nccl'),default='gloo')
    ap.add_argument('--bank',default='/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/smoke_bank')
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    world=torch.cuda.device_count() if args.backend=='nccl' else 2
    if args.backend=='nccl' and world<2: raise SystemExit('NCCL needs at least 2 GPUs')
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    mp.spawn(worker,args=(world,port,args.bank,out,args.phase,args.backend),nprocs=world,join=True)
    print(json.dumps([json.loads((out/f'{args.phase}_rank{r}.json').read_text()) for r in range(world)],indent=2))


if __name__=='__main__': main()
