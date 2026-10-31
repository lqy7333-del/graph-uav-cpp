"""Four-GPU NCCL fault injection for the all-rank pre-update gate (no training)."""
from __future__ import annotations
import json, os, socket, sys, tempfile
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.trainer.distributed import DistributedTrainer, resolve_config
from types import SimpleNamespace

def _worker(rank, world_size, directory, port):
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    try:
        config=resolve_config(SimpleNamespace(num_fields=None,set=[],ppo_epochs=None,
                                               batch_size=None,gae_lambda=None))
        agent=PPOAgent(CoveragePolicy(None,config=config),config,device=torch.device(f'cuda:{rank}'))
        trainer=DistributedTrainer(None,agent,rank=rank,world_size=world_size,
                                   checkpoint_dir=directory,config=config)
        good={'value_state_gap_max':0.0,'value_state_scale':1.0,
              'tail_value_error':0.0,'tail_value_scale':1.0,
              'replay_logprob_error':0.0,'replay_value_error':0.0,
              'episode_closure_error_max':0.0,'finite':True}
        trainer._gate_or_abort(good, 1)
        bad=dict(good); bad['value_state_gap_max']=1.0 if rank==3 else 0.0
        saw=False
        try: trainer._gate_or_abort(bad, 2)
        except RuntimeError: saw=True
        Path(directory,f'rank{rank}_nccl_fault.json').write_text(
            json.dumps({'rank':rank,'saw_failure':saw}))
        assert saw
    finally:
        dist.destroy_process_group()

def main():
    world_size=torch.cuda.device_count()
    assert world_size==4
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    directory=REPO/'coverage_env/results/maneuver_time/gate_faults_nccl'
    if directory.exists():
        import shutil; shutil.rmtree(directory)
    directory.mkdir(parents=True)
    os.environ['MASTER_ADDR']='127.0.0.1'; os.environ['MASTER_PORT']=str(port)
    mp.spawn(_worker,args=(world_size,str(directory),port),nprocs=world_size,join=True)
    report={'world_size':world_size,'rank3_injected':True,
            'ranks':[json.loads((directory/f'rank{r}_nccl_fault.json').read_text())
                     for r in range(world_size)],
            'summary_exists':(directory/'gate_failures/update_2_summary.json').exists()}
    (directory/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=1))

if __name__=='__main__': main()
