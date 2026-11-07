"""Evaluate frozen NN and 10-field A checkpoints on fixed 30-field panels."""
import copy, hashlib, json, sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.env.fixed_instance_pool import load_bank
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.evaluation.maneuver_eval import run_episode, replay_recorded, _nn_choice, _policy_choice_factory
EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode30')
OUT=EXP/'reference_eval'
MODELS={
 'NN':None,
 'A10_u1060':Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_instances_s42/ckpt_update_1060.pt'),
 'A10_u1200':Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_instances_s42/ckpt_update_1200.pt'),
}
PANELS=('diagnostic10','validation30')

def worker(rank,world):
    torch.cuda.set_device(rank)
    policies={}
    for label,path in MODELS.items():
        if path is None: continue
        ck=torch.load(path,map_location='cpu',weights_only=False); config=copy.deepcopy(ck['resolved_config'])
        template=load_bank(EXP/'diagnostic10')[1][0]
        actor=CoveragePolicy(template,config=config).cuda()
        agent=PPOAgent(actor,config,device='cuda')
        agent.load_checkpoint(path,map_location='cuda',restore_rng=False,load_optimizers=False)
        agent.actor.eval(); policies[label]=(agent,_policy_choice_factory(agent.actor))
    rows=[]; actions_out={}
    for panel_name in PANELS:
        _,envs=load_bank(EXP/panel_name)
        for i,env in enumerate(envs):
            if i%world!=rank: continue
            for label,path in MODELS.items():
                choose=_nn_choice if label=='NN' else policies[label][1]
                res=run_episode(copy.deepcopy(env),choose,record=True)
                replay=replay_recorded(copy.deepcopy(env),res['actions'])
                rec={k:res[k] for k in ('completed','termination_reason','total_time','time_work_base','time_nonwork_base','time_maneuver_extra','time_service','total_distance','coverage_ratio','steps','failed_steps','visits','resupplies','time_closure_error')}
                rec.update(panel=panel_name,index=i,seed=int(env._instance_seed) if hasattr(env,'_instance_seed') else None,
                           fingerprint=__import__('coverage_env.rl.trainer.episode',fromlist=['instance_fingerprint']).instance_fingerprint(env),
                           model=label,replay_completed=replay['completed'],
                           replay_termination_reason=replay['termination_reason'],replay_total_time=replay['total_time'],
                           replay_match=(res['completed']==replay['completed'] and res['termination_reason']==replay['termination_reason']
                                         and abs(res['total_time']-replay['total_time'])<1e-6))
                rows.append(rec); actions_out[f'{panel_name}/{label}/{i}']=res['actions']
    (OUT/f'rows_rank{rank}.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    (OUT/f'actions_rank{rank}.json').write_text(json.dumps(actions_out,ensure_ascii=False))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    world=torch.cuda.device_count(); mp.spawn(worker,args=(world,),nprocs=world,join=True)
    rows=[]
    for p in OUT.glob('rows_rank*.json'): rows+=json.loads(p.read_text())
    rows.sort(key=lambda r:(r['panel'],r['model'],r['index']))
    import csv
    keys=['panel','index','seed','fingerprint','model','completed','replay_completed','replay_match','termination_reason','total_time','replay_total_time','time_work_base','time_nonwork_base','time_maneuver_extra','time_service','total_distance','coverage_ratio','steps','failed_steps','visits','resupplies','time_closure_error']
    with (OUT/'rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})
    summary={'models':{},'panels':{}}
    for panel in PANELS:
        for model in MODELS:
            rr=[r for r in rows if r['panel']==panel and r['model']==model]
            summary['panels'].setdefault(panel,{})[model]={'count':len(rr),'completed':sum(bool(r['completed']) for r in rr),
                'mean':sum(float(r['total_time']) for r in rr)/len(rr),
                'components':{k:sum(float(r[k]) for r in rr)/len(rr) for k in ('time_work_base','time_nonwork_base','time_maneuver_extra','time_service')},
                'distance':sum(float(r['total_distance']) for r in rr)/len(rr),
                'visits':sum(float(r['visits']) for r in rr)/len(rr),'resupplies':sum(float(r['resupplies']) for r in rr)/len(rr),
                'replay_all_match':all(bool(r['replay_match']) for r in rr)}
    summary['checkpoints']={k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in MODELS.items() if v is not None}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
