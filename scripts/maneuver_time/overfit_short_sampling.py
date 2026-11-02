"""8-sample endpoint check for the 60-update actor-LR screen."""
from __future__ import annotations
import copy, csv, json, math, statistics, sys
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp

REPO=Path('/home/ps/Desktop/Graph_planning/graph-uav-overfit-persist')
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from coverage_env.env.fixed_instance_pool import load_bank
from coverage_env.rl.decoder.action_decoder import CoveragePolicy
from coverage_env.rl.evaluation.maneuver_eval import run_episode
from coverage_env.rl.ppo.ppo import PPOAgent
from coverage_env.rl.checkpoint import state_dict_sha256

EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/overfit20')
OUT=EXP/'short'
BANK=EXP/'bank'
RUNS={'control':OUT/'control_s42','low':OUT/'lr3e5_s42'}
REPS=8
SEED_BASE=910000
POINTS={'start':('control','ckpt_update_0.pt'),
        'control60':('control','ckpt_update_60.pt'),
        'low60':('low','ckpt_update_60.pt')}

def choose_sampled(policy):
    def choose(env,state):
        actions=env.get_action_candidates(); mask=env.get_action_mask(actions)
        with torch.no_grad():
            out=policy(state,environment=env,candidate_actions=actions,feasible_mask=mask)
            dist=out['action_distribution']; idx=int(torch.multinomial(dist,1).item())
        return out['actions'][idx],0.,0.
    return choose

def worker(rank,world):
    torch.set_num_threads(1); torch.cuda.set_device(rank)
    device=torch.device(f'cuda:{rank}')
    manifest,templates=load_bank(BANK)
    rows=[]
    for label,(run,file) in POINTS.items():
        ck=torch.load(RUNS[run]/file,map_location='cpu',weights_only=False)
        config=copy.deepcopy(ck['resolved_config'])
        agent=PPOAgent(CoveragePolicy(templates[0],config=config),config,device=device)
        agent.load_checkpoint(RUNS[run]/file,map_location=device,restore_rng=False,load_optimizers=False)
        agent.actor.eval(); choose=choose_sampled(agent.actor)
        for i in range(10):
            for rep in range(REPS):
                task=i*REPS+rep
                if task%world!=rank: continue
                seed=SEED_BASE+task
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                env=copy.deepcopy(templates[i]); result=run_episode(env,choose)
                rows.append({'rank':rank,'label':label,'index':i,'seed':manifest['instances'][i]['seed'],
                             'replicate':rep,**{k:result[k] for k in ('completed','total_time','total_distance','time_work_base','time_nonwork_base','time_maneuver_extra','time_service','coverage_ratio','visits','resupplies')}})
        print(f'[rank {rank}] {label} done',flush=True)
    (OUT/f'sampled_rank_{rank}.json').write_text(json.dumps(rows))

def cond_ci(vals,samples=4000,seed=5):
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(samples):
        allv=[]
        for i in range(10):
            a=np.asarray(vals[i]); allv.extend(a[rng.integers(0,len(a),len(a))].tolist())
        draws.append(float(np.mean(allv)))
    return [float(x) for x in np.percentile(draws,[2.5,97.5])]

def main():
    if len(list(OUT.glob('sampled_rank_*.json'))) != 4:
        for p in OUT.glob('sampled_rank_*.json'): p.unlink()
        mp.spawn(worker,args=(4,),nprocs=4,join=True)
    rows=[]
    for p in sorted(OUT.glob('sampled_rank_*.json')): rows+=json.loads(p.read_text())
    rows.sort(key=lambda r:(r['label'],r['index'],r['replicate']))
    assert len(rows)==3*10*REPS
    by={}
    for r in rows: by.setdefault((r['label'],r['index']),[]).append(float(r['total_time']))
    greedy={}
    for label in POINTS:
        run,file=POINTS[label]
        u=int(file.split('_')[-1].split('.')[0])
        data=json.loads((RUNS[run]/f'evaluation/update_{u}.json').read_text())
        greedy[label]={r['index']:float(r['total_time']) for r in data['rows']}
    summary={'replicates':REPS,'points':{},'greedy_curve':{},'actor_distance':{}}
    for label in POINTS:
        vals={i:by[(label,i)] for i in range(10)}
        mean=float(np.mean([np.mean(v) for v in vals.values()]))
        summary['points'][label]={'sampled_mean':mean,'conditional_ci95':cond_ci(vals),
            'per_instance':{str(i):float(np.mean(v)) for i,v in vals.items()},
            'completed':sum(bool(r['completed']) for r in rows if r['label']==label),
            'samples':sum(1 for r in rows if r['label']==label)}
    for label in ('control','low'):
        rows2=list(csv.DictReader((RUNS[label]/'learning_curve.csv').open()))
        summary['greedy_curve'][label]=[{'update':int(r['update']),'total_time':float(r['total_time']),
            'initial_minus':float(r['initial_minus_candidate_total_time']),
            'nn_minus':float(r['nn_minus_candidate_total_time']),'completed':int(r['completed'])} for r in rows2]
        last=rows2[-1]; print(label,'last',last['update'],last['total_time'])
    start=torch.load(RUNS['control']/'ckpt_update_0.pt',map_location='cpu',weights_only=False)
    for label,file in (('control60','ckpt_update_60.pt'),('low60','ckpt_update_60.pt')):
        end=torch.load(RUNS['control' if label=='control60' else 'low']/file,map_location='cpu',weights_only=False)
        diffs=[]; base=[]
        for k in start['actor']:
            diffs.append((end['actor'][k].float()-start['actor'][k].float()).flatten())
            base.append(start['actor'][k].float().flatten())
        d=torch.cat(diffs); b=torch.cat(base)
        summary['actor_distance'][label]={'l2':float(d.norm().item()),'relative_l2':float((d.norm()/b.norm()).item()),
            'max_abs':float(d.abs().max().item()),'actor_sha':state_dict_sha256(end['actor'])}
    (OUT/'sampled_endpoints.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    with (OUT/'sampled_endpoints.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['label','index','seed','sampled_mean','n'])
        for label in POINTS:
            for i in range(10):
                vals=by[(label,i)]
                w.writerow([label,i,70000+i,statistics.mean(vals),len(vals)])
    print(json.dumps(summary['points'],indent=2))

if __name__=='__main__': main()
