"""Two-rank Gloo failure injection for complete-episode PPO.

Run with RUN_DDP_TESTS=1; writes a JSON result and diagnostics, never trains.
"""
import argparse, copy, datetime, json, os, socket, sys, tempfile, traceback
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.config import load_config
from coverage_env.env.fixed_instance_pool import FixedInstancePool, write_bank
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.episode_rollout import EpisodeRolloutTrainer


def _worker(rank, world, port, bank, base, phase):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    try:
        config=json.loads((Path(bank)/'manifest.json').read_text())['config']
        config['ppo']['collection_mode']='complete_episodes'
        config['ppo']['episodes_per_rank']=1
        config['ppo']['batch_size']=2
        config['ppo']['epochs']=1
        torch.manual_seed(42)
        pool=FixedInstancePool(bank, seed=42+rank)
        agent=PPOAgent(CoveragePolicy(pool, config=config), config, device='cpu')
        out=Path(base)/phase
        trainer=EpisodeRolloutTrainer(pool, agent, rank=rank, world_size=world,
                                      checkpoint_dir=out, checkpoint_every=1,
                                      config=config, episodes_per_rank=1,
                                      max_episode_steps=50)
        if rank==1 and phase=='collect':
            trainer._collect_episodes=lambda: (_ for _ in ()).throw(RuntimeError('injected collect failure'))
        if rank==1 and phase=='gate':
            trainer._local_gate_metrics=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('injected gate failure'))
        result={'rank':rank,'phase':phase,'raised':False,'error':'','update_count':None,
                'log_exists':(out/'train_log.csv').exists(),'diag_exists':False}
        try:
            trainer.train(1, rollout_steps=1)
        except Exception as exc:
            result['raised']=True; result['error']=repr(exc)
        result['update_count']=int(getattr(agent,'update_count',-1))
        result['log_exists']=(out/'train_log.csv').exists()
        diag=out/'run_failures'/f'update_1_{"gate_construct" if phase=="gate" else phase}.json'
        result['diag_exists']=diag.exists()
        (Path(base)/f'{phase}_rank{rank}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--phase',choices=('collect','gate'),required=True)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    bank=out/'bank'
    if not bank.exists():
        c=load_config(); c['environment']['num_fields']={'min':2,'max':2}; c['field_generator']['layout']={'mode':'scattered'}; c['field_generator']['field_size']={'min_area':500.,'max_area':800.}; c['ppo'].update(epochs=1,batch_size=2,value_gate=True,randomize_uav=False,gamma=1.); c['ppo']['collection_mode']='complete_episodes'; c['ppo']['episodes_per_rank']=1
        envs=[InstanceGenerator(c).generate(num_fields=2,seed=s,randomize_each_reset=False,randomize_uav=False) for s in (70000,70001)]
        write_bank(bank,c,envs,[70000,70001])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    mp.spawn(_worker,args=(2,port,bank,out,args.phase),nprocs=2,join=True)
    rows=[json.loads((out/f'{args.phase}_rank{r}.json').read_text()) for r in (0,1)]
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
