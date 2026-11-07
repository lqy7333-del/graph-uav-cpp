"""Evaluate frozen C/D best/final on the existing A/B fresh 10-field panel."""
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.evaluation.maneuver_eval import run_episode, _policy_choice_factory
EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10')
PANEL=EXP/'ab_fresh_confirm_s42/instances.pt'
OUT=EXP/'ab_cd_fresh_supplement_s42'
MODELS={
 'C_best':EXP/'ablation_no_inter_s42/ckpt_update_1020.pt',
 'C_final':EXP/'ablation_no_inter_s42/ckpt_update_1200.pt',
 'D_best':EXP/'ablation_no_intra_s42/ckpt_update_900.pt',
 'D_final':EXP/'ablation_no_intra_s42/ckpt_update_1200.pt',
}

def worker(rank,world,panel_path):
    torch.cuda.set_device(rank)
    saved=torch.load(panel_path,map_location='cpu',weights_only=False)
    template=saved['instances'][0]
    models={}
    for label,path in MODELS.items():
        ck=torch.load(path,map_location='cpu',weights_only=False); config=copy.deepcopy(ck['resolved_config'])
        actor=CoveragePolicy(template,config=config).cuda()
        agent=PPOAgent(actor,config,device='cuda')
        agent.load_checkpoint(path,map_location='cuda',restore_rng=False,load_optimizers=False)
        agent.actor.eval(); models[label]=_policy_choice_factory(agent.actor)
    rows=[]
    for i,env in enumerate(saved['instances']):
        if i%world!=rank: continue
        for label,choose in models.items():
            row=run_episode(copy.deepcopy(env),choose)
            row.update(model=label,index=i,seed=saved['seeds'][i],fingerprint=saved['fingerprints'][i])
            rows.append(row)
    (OUT/f'rows_rank{rank}.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    saved=torch.load(PANEL,map_location='cpu',weights_only=False)
    if len(saved['instances'])!=20: raise RuntimeError('expected 20 instances')
    world=torch.cuda.device_count(); mp.spawn(worker,args=(world,PANEL),nprocs=world,join=True)
    rows=[]
    for p in OUT.glob('rows_rank*.json'): rows+=json.loads(p.read_text())
    rows.sort(key=lambda r:(r['index'],r['model']))
    import csv
    keys=['index','seed','fingerprint','model','completed','termination_reason','total_time','time_work_base','time_nonwork_base','time_maneuver_extra','time_service','total_distance','coverage_ratio','steps','failed_steps']
    with (OUT/'rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})
    (OUT/'summary.json').write_text(json.dumps({'panel':str(PANEL),'models':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in MODELS.items()}},indent=2))
    print('wrote',len(rows),'rows')
if __name__=='__main__': main()
