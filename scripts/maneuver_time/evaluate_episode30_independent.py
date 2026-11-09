"""One-shot independent 30-field evaluation of frozen new30/A10/NN models."""
import copy, hashlib, json, sys
from pathlib import Path
import torch
import torch.multiprocessing as mp
ROOT=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from coverage_env.env.instance_generator import InstanceGenerator, GeometryGenerationError
from coverage_env.env.fixed_instance_pool import write_bank, load_bank
from coverage_env.rl.trainer.episode import instance_fingerprint
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.evaluation.maneuver_eval import run_episode, replay_recorded, _nn_choice, _policy_choice_factory

EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode30')
OUT=EXP/'independent_eval_s42'
PANEL=EXP/'independent30'
BASE=1_340_000; N=30
MODELS={
 'new30_best':EXP/'formal_random_instances_s42/ckpt_update_1060.pt',
 'new30_final':EXP/'formal_random_instances_s42/ckpt_update_1200.pt',
 'A10_best':Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_instances_s42/ckpt_update_1060.pt'),
 'A10_final':Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/episode10/formal_random_instances_s42/ckpt_update_1200.pt'),
 'NN':None,
}

def build_panel():
    if (PANEL/'manifest.json').exists():
        raise FileExistsError('independent panel already exists; do not rebuild')
    ck=torch.load(MODELS['new30_best'],map_location='cpu',weights_only=False)
    config=copy.deepcopy(ck['resolved_config']); config['environment']['num_fields']={'min':30,'max':30}
    config['environment']['num_fields_strategy']='uniform'
    gen=InstanceGenerator(config); envs=[]; seeds=[]; failures=[]; seed=BASE
    while len(envs)<N:
        try:
            env=gen.generate(num_fields=30,seed=seed,randomize_each_reset=False,randomize_uav=False)
            if len(env.fields)!=30: raise RuntimeError('not 30 fields')
            envs.append(env); seeds.append(seed)
        except GeometryGenerationError as exc:
            failures.append({'seed':seed,'internal_attempts':8,'error':str(exc)})
        seed+=1
    manifest=write_bank(PANEL,config,envs,seeds)
    log={'base_seed':BASE,'accepted':len(envs),'rejected':len(failures),'failures':failures,
         'seeds':seeds,'fingerprints':[instance_fingerprint(e) for e in envs],
         'manifest_sha256':hashlib.sha256((PANEL/'manifest.json').read_bytes()).hexdigest(),
         'instances_sha256':manifest['bank_sha256'],'config':config}
    (EXP/'independent_panel_log.json').write_text(json.dumps(log,indent=2,ensure_ascii=False))
    return log

def worker(rank,world,out):
    torch.cuda.set_device(rank)
    manifest,envs=load_bank(PANEL)
    seeds=[e['seed'] for e in manifest['instances']]; fps=[e['fingerprint'] for e in manifest['instances']]
    choices={}
    for label,path in MODELS.items():
        if path is None: continue
        ck=torch.load(path,map_location='cpu',weights_only=False); config=copy.deepcopy(ck['resolved_config'])
        actor=CoveragePolicy(envs[0],config=config).cuda()
        agent=PPOAgent(actor,config,device='cuda')
        agent.load_checkpoint(path,map_location='cuda',restore_rng=False,load_optimizers=False)
        agent.actor.eval(); choices[label]=_policy_choice_factory(agent.actor)
    rows=[]; actions_out={}
    for i,env in enumerate(envs):
        if i%world!=rank: continue
        for label,path in MODELS.items():
            choose=_nn_choice if label=='NN' else choices[label]
            res=run_episode(copy.deepcopy(env),choose,record=True)
            rep=replay_recorded(copy.deepcopy(env),res['actions'])
            r={k:res[k] for k in ('completed','termination_reason','total_time','time_work_base','time_nonwork_base','time_maneuver_extra','time_service','total_distance','coverage_ratio','steps','failed_steps','visits','resupplies','time_closure_error')}
            r.update(index=i,seed=seeds[i],fingerprint=fps[i],model=label,replay_ok=bool(rep['completed']==res['completed'] and rep['termination_reason']==res['termination_reason'] and abs(rep['total_time']-res['total_time'])<1e-6))
            rows.append(r); actions_out[f'{label}/{i}']=res['actions']
    (out/f'rows_rank{rank}.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    (out/f'actions_rank{rank}.json').write_text(json.dumps(actions_out,ensure_ascii=False))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (PANEL/'manifest.json').exists(): print(build_panel())
    world=torch.cuda.device_count(); mp.spawn(worker,args=(world,OUT),nprocs=world,join=True)
    rows=[]
    for p in OUT.glob('rows_rank*.json'): rows+=json.loads(p.read_text())
    rows.sort(key=lambda r:(r['model'],r['index']))
    import csv
    keys=['index','seed','fingerprint','model','completed','replay_ok','termination_reason','total_time','time_work_base','time_nonwork_base','time_maneuver_extra','time_service','total_distance','coverage_ratio','steps','failed_steps','visits','resupplies','time_closure_error']
    with (OUT/'rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in keys})
    by={m:{r['index']:r for r in rows if r['model']==m} for m in MODELS}
    import statistics,numpy as np
    def boot(d):
        rng=np.random.default_rng(20261109); vals=[float(np.mean(rng.choice(d,len(d),replace=True))) for _ in range(10000)]
        return [float(x) for x in np.percentile(vals,[2.5,97.5])]
    summary={'panel':str(PANEL),'instances':N,'seeds':[by['NN'][i]['seed'] for i in range(N)],
             'models':{m:{'mean':statistics.mean(float(r['total_time']) for r in by[m].values()),
                           'completed':sum(bool(r['completed']) for r in by[m].values()),
                           'components':{k:statistics.mean(float(r[k]) for r in by[m].values()) for k in ('time_work_base','time_nonwork_base','time_maneuver_extra','time_service')},
                           'visits':statistics.mean(float(r['visits']) for r in by[m].values()),
                           'resupplies':statistics.mean(float(r['resupplies']) for r in by[m].values()),
                           'distance':statistics.mean(float(r['total_distance']) for r in by[m].values()),
                           'replay_all':all(bool(r['replay_ok']) for r in by[m].values())} for m in MODELS},
             'pairs':{}}
    for key,base,cand in (('new_best_vs_A10_best','A10_best','new30_best'),('new_final_vs_A10_final','A10_final','new30_final'),
                          ('new_best_vs_A10_final','A10_final','new30_best'),('new_final_vs_A10_best','A10_best','new30_final'),
                          ('new_best_vs_NN','NN','new30_best'),('new_final_vs_NN','NN','new30_final')):
        dif=[float(by[base][i]['total_time'])-float(by[cand][i]['total_time']) for i in range(N)]
        summary['pairs'][key]={'baseline':base,'candidate':cand,'baseline_minus_candidate':float(np.mean(dif)),
          'ci95_instance_bootstrap':boot(dif),'new_wins':sum(x>1e-6 for x in dif),'baseline_wins':sum(x<-1e-6 for x in dif),'ties':sum(abs(x)<=1e-6 for x in dif),
          'per_instance':[{'index':i,'seed':by['NN'][i]['seed'],'baseline_time':float(by[base][i]['total_time']),'candidate_time':float(by[cand][i]['total_time']),'difference':dif[i]} for i in range(N)]}
    summary['checkpoints']={k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in MODELS.items() if v}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps({'models':{m:round(v['mean'],1) for m,v in summary['models'].items()},'pairs':{k:{kk:round(vv,2) if isinstance(vv,float) else vv for kk,vv in v.items() if kk!='per_instance'} for k,v in summary['pairs'].items()}},indent=2))
if __name__=='__main__': main()
