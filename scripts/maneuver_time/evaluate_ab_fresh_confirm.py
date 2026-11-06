"""Paired fresh 10-field confirmation of A/B selected and final checkpoints."""
import copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.env.instance_generator import InstanceGenerator
from coverage_env.rl.trainer.episode import instance_fingerprint
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.evaluation.maneuver_eval import run_episode, _nn_choice, _policy_choice_factory

EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10')
OUT=EXP/'ab_fresh_confirm_s42'
MODELS={
 'A_best':EXP/'formal_random_instances_s42/ckpt_update_1060.pt',
 'A_final':EXP/'formal_random_instances_s42/ckpt_update_1200.pt',
 'B_best':EXP/'ablation_node_mlp_s42/ckpt_update_760.pt',
 'B_final':EXP/'ablation_node_mlp_s42/ckpt_update_1200.pt',
}
SEED_BASE=1_100_000
N=20


def _load_models(template):
    models={}
    for label,path in MODELS.items():
        ck=torch.load(path,map_location='cpu',weights_only=False)
        config=copy.deepcopy(ck['resolved_config'])
        actor=CoveragePolicy(template,config=config)
        agent=PPOAgent(actor,config,device='cuda')
        agent.load_checkpoint(path,map_location='cuda',restore_rng=False,load_optimizers=False)
        agent.actor.eval()
        models[label]=(config,agent.actor,_policy_choice_factory(agent.actor))
    return models


def worker(rank,world,instances_path):
    torch.cuda.set_device(rank)
    saved=torch.load(instances_path,map_location='cpu',weights_only=False)
    template=saved['instances'][0]
    models=_load_models(template)
    rows=[]
    for i,env in enumerate(saved['instances']):
        if i%world!=rank: continue
        seed=saved['seeds'][i]; fp=saved['fingerprints'][i]
        for label,(config,actor,choose) in models.items():
            row=run_episode(copy.deepcopy(env),choose)
            row.update(model=label,index=i,seed=seed,fingerprint=fp)
            rows.append(row)
        n=run_episode(copy.deepcopy(env),_nn_choice)
        n.update(model='NN',index=i,seed=seed,fingerprint=fp)
        rows.append(n)
    (OUT/f'rows_rank{rank}.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))


def boot(diffs):
    rng=np.random.default_rng(20261106); vals=[]
    for _ in range(10000):
        vals.append(float(np.mean(rng.choice(diffs,len(diffs),replace=True))))
    return [float(x) for x in np.percentile(vals,[2.5,97.5])]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ck=torch.load(MODELS['A_best'],map_location='cpu',weights_only=False)
    config=copy.deepcopy(ck['resolved_config']); config['environment']['num_fields']={'min':10,'max':10}
    envs=[]; seeds=[]
    for i in range(N):
        seed=SEED_BASE+i
        env=InstanceGenerator(config).generate(num_fields=10,seed=seed,randomize_each_reset=False,randomize_uav=False)
        if len(env.fields)!=10: raise RuntimeError('field count mismatch')
        envs.append(env); seeds.append(seed)
    fps=[instance_fingerprint(e) for e in envs]
    torch.save({'instances':envs,'seeds':seeds,'fingerprints':fps,'config':config,
                'checkpoints':{k:str(v) for k,v in MODELS.items()}},OUT/'instances.pt')
    world=torch.cuda.device_count()
    mp.spawn(worker,args=(world,OUT/'instances.pt'),nprocs=world,join=True)
    rows=[]
    for p in OUT.glob('rows_rank*.json'): rows+=json.loads(p.read_text())
    rows.sort(key=lambda r:(r['index'],r['model']))
    import csv
    keys=['index','seed','fingerprint','model','completed','termination_reason','total_time','time_work_base','time_nonwork_base','time_maneuver_extra','time_service','total_distance','coverage_ratio','steps','failed_steps']
    with (OUT/'rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})
    by={m:{r['index']:r for r in rows if r['model']==m} for m in list(MODELS)+['NN']}
    assert all(len(by[m])==N for m in by)
    out={'checkpoints':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest(),
                            'update':torch.load(v,map_location='cpu',weights_only=False)['update_count']}
                        for k,v in MODELS.items()},
         'instances':N,'field_count':10,'seed_base':SEED_BASE,'seeds':seeds,'fingerprints':fps,
         'models':{},'pairs':{}}
    for m in by:
        rr=list(by[m].values()); out['models'][m]={'completed':sum(r['completed'] for r in rr),
            'mean_all':sum(r['total_time'] for r in rr)/N,
            'mean_complete':float(np.mean([r['total_time'] for r in rr if r['completed']])),
            'time_work_base':float(np.mean([r['time_work_base'] for r in rr])),
            'time_nonwork_base':float(np.mean([r['time_nonwork_base'] for r in rr])),
            'time_maneuver_extra':float(np.mean([r['time_maneuver_extra'] for r in rr])),
            'time_service':float(np.mean([r['time_service'] for r in rr]))}
    for name,a,b in (('selected','A_best','B_best'),('final','A_final','B_final'),
                     ('A_best_vs_A_final','A_best','A_final'),('B_best_vs_B_final','B_best','B_final')):
        dif=[by[b][i]['total_time']-by[a][i]['total_time'] for i in range(N)]
        out['pairs'][name]={'A':a,'B':b,'B_minus_A':float(np.mean(dif)),
            'B_minus_A_ci95_fixed_instances':boot(dif),
            'A_better_count':sum(d>1e-6 for d in dif),'B_better_count':sum(d<-1e-6 for d in dif),
            'ties':sum(abs(d)<=1e-6 for d in dif),
            'per_instance':[{'index':i,'seed':seeds[i],
                             'A_time':by[a][i]['total_time'],'B_time':by[b][i]['total_time'],
                             'B_minus_A':dif[i]} for i in range(N)]}
    (OUT/'summary.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps({'models':{k:round(v['mean_all'],1) for k,v in out['models'].items()},
                      'pairs':{k:{kk:round(vv,2) if isinstance(vv,float) else vv for kk,vv in v.items() if kk!='per_instance'} for k,v in out['pairs'].items()}},indent=2))


if __name__=='__main__': main()
