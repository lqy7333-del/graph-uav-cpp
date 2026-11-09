"""Summarize the 5-instance x 3-seed nested-GA pilot and same-snapshot controls."""
import csv, hashlib, json, platform, statistics
from pathlib import Path
import numpy as np

EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments')
PILOT=EXP/'nested_ga/pilot10_s1400000'; CTRL=EXP/'nested_ga/controls10_s1400000'; OUT=EXP/'nested_ga/report10_s1400000'
OUT.mkdir(parents=True,exist_ok=True)
trials=[]; curves=[]
for path in sorted(PILOT.glob('instance_*_ga_*.json')):
    d=json.loads(path.read_text()); best=d['best']; q=d['inner_quotas']; hist=d['history']
    full=sum(1 for x in q if x['evaluations']>=16)
    partial=sum(1 for x in q if 0 < x['evaluations']<16)
    empty=sum(1 for x in q if x['evaluations']==0)
    total_evals=d['evaluations']; incomplete=sum(not x['completed'] for x in hist)
    trial={
      'instance_seed':d['instance_seed'],'search_seed':d['config']['seed'],'snapshot_sha256':d['instance_snapshot_sha256'],
      'best_cost':best['cost'],'baseline_cost':d['baseline']['cost'],'improvement':d['baseline']['cost']-best['cost'],
      'completed':bool(best['completed']),'reason':best['reason'],'termination':d['termination'],
      'evaluations':total_evals,'transitions':d['transitions'],'search_seconds':d['search_seconds'],
      'replay_seconds':d['replay_seconds'],'total_seconds':d['search_seconds']+d['replay_seconds'],
      'quotas':len(q),'full_quotas':full,'partial_quotas':partial,'empty_quotas':empty,
      'quotas_completed':sum(1 for x in q if x['completed']),
      'infeasible_candidates':incomplete,'infeasible_ratio':incomplete/total_evals,
      'fallback_steps':best['fallback_steps'],'skipped_slots':best['skipped_slots'],
      'execution_steps':best['summary']['execution_count'],
      'fallback_ratio':best['fallback_steps']/best['summary']['execution_count'] if best['summary']['execution_count'] else None,
      'total_time':best['summary']['total_time'],'work':best['summary']['time_work_base'],
      'nonwork':best['summary']['time_nonwork_base'],'maneuver':best['summary']['time_maneuver_extra'],
      'service':best['summary']['time_service'],'distance':best['summary']['total_distance'],
      'visits':best['summary']['execution_count'],'resupplies':best['summary']['resupply_count'],
      'closure':best['summary']['time_closure_error']
    }
    trials.append(trial)
    # progress at each evaluation: incumbent and elapsed search time
    for h in hist:
        curves.append({'instance_seed':d['instance_seed'],'search_seed':d['config']['seed'],
                       'evaluation':h['evaluation'],'seconds':h['seconds'],'incumbent':h['best'],
                       'completed':h['completed'],'cost':h['cost']})
controls=list(csv.DictReader((CTRL/'rows.csv').open()))
def cmean(model):
    rr=[r for r in controls if r['model']==model]; assert len(rr)==5
    return {k:sum(float(r[k]) for r in rr)/len(rr) for k in
            ('total_time','time_work_base','time_nonwork_base','time_maneuver_extra','time_service','total_distance','visits','resupplies','eval_wall_seconds')} | {'completed':sum(str(r['completed']).lower()=='true' for r in rr),'replay_all':all(str(r['replay_ok']).lower()=='true' for r in rr)}
models=['original_NN','ga_initializer','A10_u1060','A10_u1200']
per=[]
for seed in sorted(set(t['instance_seed'] for t in trials)):
    ts=[t for t in trials if t['instance_seed']==seed]; assert len(ts)==3
    vals=[t['best_cost'] for t in ts]
    row={'instance_seed':seed,'ga_mean':statistics.mean(vals),'ga_std':statistics.stdev(vals),'ga_min':min(vals),'ga_max':max(vals),'ga_best_of_3':min(vals)}
    for model in models:
        rr=[r for r in controls if int(r['instance_seed'])==seed and r['model']==model][0]
        row[model+'_time']=float(rr['total_time']); row[model+'_minus_ga_mean']=float(rr['total_time'])-row['ga_mean']
        row[model+'_minus_ga_best3']=float(rr['total_time'])-row['ga_best_of_3']
    row['ga_mean_nonwork_maneuver_service']=statistics.mean(t['nonwork']+t['maneuver']+t['service'] for t in ts)
    for model in models:
        rr=[r for r in controls if int(r['instance_seed'])==seed and r['model']==model][0]
        row[model+'_nonwork_maneuver_service']=float(rr['time_nonwork_base'])+float(rr['time_maneuver_extra'])+float(rr['time_service'])
    per.append(row)
# CSV outputs
keys=list(trials[0])
with (OUT/'ga_trials.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(trials)
pkeys=['instance_seed','ga_mean','ga_std','ga_min','ga_max','ga_best_of_3','ga_mean_nonwork_maneuver_service']+[f'{m}_{s}' for m in models for s in ('time','minus_ga_mean','minus_ga_best3','nonwork_maneuver_service')]
with (OUT/'per_instance.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=pkeys); w.writeheader(); w.writerows(per)
with (OUT/'progress.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['instance_seed','search_seed','evaluation','seconds','incumbent','completed','cost']); w.writeheader(); w.writerows(curves)
summary={'pilot':str(PILOT),'controls':str(CTRL),'instances':5,'search_seeds':[42,43,44],'searches':len(trials),
 'controls_mean':{m:cmean(m) for m in models},'trials':trials,
 'search':{k: {'mean':statistics.mean(t[k] for t in trials),'min':min(t[k] for t in trials),'max':max(t[k] for t in trials)} for k in ('evaluations','transitions','search_seconds','replay_seconds','total_seconds','full_quotas','partial_quotas','infeasible_candidates','infeasible_ratio','fallback_steps','fallback_ratio','skipped_slots','improvement')},
 'per_instance':per}
for m in models:
    ds=[r[m+'_minus_ga_mean'] for r in per]
    summary.setdefault('control_minus_ga_mean',{})[m]={'mean':statistics.mean(ds),'min':min(ds),'max':max(ds),
       'ga_faster_instances':sum(x>1e-6 for x in ds)}
for r in per:
    r['ga_best_of_3_note']='3x per-instance search budget; not a single-search result'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps({'search':summary['search'],'controls_mean':{m:round(v['total_time'],1) for m,v in summary['controls_mean'].items()},'control_minus_ga_mean':{m:round(v['mean'],1) for m,v in summary['control_minus_ga_mean'].items()}},indent=2))
