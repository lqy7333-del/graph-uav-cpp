"""Summarize 15/20-field seed42 nested-GA comparisons separately; 4-test Holm."""
import csv,json,statistics
from pathlib import Path
import numpy as np
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga')
OUT=BASE/'scale_report_seed42'; OUT.mkdir(parents=True,exist_ok=True)
scales=['independent15','independent20']; models=['original_NN','ga_initializer','A10_u1060','A10_u1200']
all_primary=[]; result={}
for scale in scales:
 panel=json.loads((BASE/scale/'panel/manifest.json').read_text()); seeds=[int(x['seed']) for x in panel['instances']]
 controls=list(csv.DictReader((BASE/scale/'controls/rows.csv').open())); cby={(int(r['instance_seed']),r['model']):r for r in controls}
 trials=[]; per=[]; progress=[]
 for seed in seeds:
  d=json.loads((BASE/scale/'runs'/f'instance_{seed}_ga_42.json').read_text()); b=d['best']; q=d['inner_quotas']
  full=sum(x['completed'] for x in q); part=sum(not x['completed'] for x in q)
  trial={'instance_seed':seed,'search_seed':42,'fields':panel['instances'][0]['field_count'],
   'snapshot_sha256':d.get('instance_snapshot_sha256',d.get('snapshot_sha256')),'task_completed':bool(b['completed']),
   'stop_reason':d['termination'],'cost':b['cost'],'total_time':b['summary']['total_time'],
   'work':b['summary']['time_work_base'],'nonwork':b['summary']['time_nonwork_base'],
   'maneuver':b['summary']['time_maneuver_extra'],'service':b['summary']['time_service'],
   'distance':b['summary']['total_distance'],'visits':b['summary']['execution_count'],'resupplies':b['summary']['resupply_count'],
   'coverage':b['summary']['coverage_ratio'],'evaluations':d['evaluations'],'transitions':d['transitions'],
   'search_seconds':d['search_seconds'],'replay_seconds':d['replay_seconds'],'per_candidate_seconds':d['search_seconds']/d['evaluations'],
   'outer_rounds':d.get('outer_generations_completed'),'outer_candidates':d.get('outer_candidates_attempted'),
   'full_quotas':full,'partial_quotas':part,'planned_search_completed':bool(d.get('outer_generations_completed')==6 and full==24 and part==0),
   'fallback_steps':b.get('fallback_steps',0),'fallback_sources':b.get('fallback_sources',{}),
   'fallback_ratio':b.get('fallback_steps',0)/b['summary']['execution_count'],
   'refresh_events':sum(bool(h.get('improved')) for h in d['history']),
   'baseline_cost':d['baseline']['cost'],'improvement':d['baseline']['cost']-b['cost']}
  trials.append(trial)
  row={k:trial[k] for k in ('instance_seed','cost','task_completed','stop_reason','planned_search_completed','evaluations','outer_rounds','full_quotas','partial_quotas','fallback_steps','fallback_sources','per_candidate_seconds')}
  row['ga_deducted']=trial['nonwork']+trial['maneuver']+trial['service']
  for m in models:
   c=cby[(seed,m)]; row[m]=float(c['total_time']); row[m+'_completed']=str(c['completed']).lower()=='true'
   row[m+'_minus_ga']=float(c['total_time'])-trial['cost']
   row[m+'_deducted']=float(c['time_nonwork_base'])+float(c['time_maneuver_extra'])+float(c['time_service'])
  per.append(row)
 for h in []: pass
 # progress
 for seed in seeds:
  d=json.loads((BASE/scale/'runs'/f'instance_{seed}_ga_42.json').read_text())
  for h in d['history']:
   progress.append({'instance_seed':seed,'evaluation':h['evaluation'],'seconds':h['seconds'],'incumbent':h['best'],
    'completed':h['completed'],'cost':h['cost'],'label':h['label'],'generation':h.get('generation'),
    'candidate':h.get('candidate'),'improved':h.get('improved')})
 def boot(d):
  rng=np.random.default_rng(20261110); arr=np.asarray(d,float); vals=[]
  for _ in range(10000): vals.append(float(np.mean(rng.choice(arr,len(arr),replace=True))))
  return np.asarray(vals)
 def comp(name,d,denom):
  arr=np.asarray(d,float); bs=boot(d); lo,hi=np.percentile(bs,[2.5,97.5])
  p=(1+min(int(np.sum(bs<=0)),int(np.sum(bs>=0))))/(len(bs)+1); p=min(1.0,2*p)
  return {'name':name,'mean':float(arr.mean()),'median':float(np.median(arr)),'ci95':[float(lo),float(hi)],
   'wins':int(np.sum(arr>1e-6)),'ties':int(np.sum(np.abs(arr)<=1e-6)),'losses':int(np.sum(arr<-1e-6)),
   'p_boot':float(p),'relative_percent':float(arr.mean()/denom*100)}
 primary=[comp('NN - GA',[r['original_NN']-r['ga'] for r in per],statistics.mean(r['original_NN'] for r in [])) if False else comp('NN - GA',[r['original_NN']-r['cost'] for r in per],statistics.mean(r['original_NN'] for r in per)),
           comp('GA - A10_u1060',[r['cost']-r['A10_u1060'] for r in per],statistics.mean(r['A10_u1060'] for r in per))]
 aux=[comp('A10_u1060 - NN',[r['A10_u1060']-r['original_NN'] for r in per],statistics.mean(r['original_NN'] for r in per)),
      comp('init - GA',[r['ga_initializer']-r['cost'] for r in per],statistics.mean(r['ga_initializer'] for r in per)),
      comp('GA - A10_u1200',[r['cost']-r['A10_u1200'] for r in per],statistics.mean(r['A10_u1200'] for r in per))]
 for x in primary: x['scale']=int(panel['instances'][0]['field_count']); all_primary.append(x)
 keys=list(trials[0])
 with (OUT/f'{scale}_ga_trials.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(trials)
 with (OUT/f'{scale}_per_instance.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(per[0])); w.writeheader(); w.writerows(per)
 with (OUT/f'{scale}_progress.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(progress[0])); w.writeheader(); w.writerows(progress)
 result[scale]={'fields':int(panel['instances'][0]['field_count']),'panel':str(BASE/scale/'panel'),
  'trials':trials,'per_instance':per,'progress':progress,'primary':primary,'aux':aux,
  'ga_mean':statistics.mean(t['cost'] for t in trials),'ga_deducted':statistics.mean(t['nonwork']+t['maneuver']+t['service'] for t in trials),
  'controls_mean':{m:{'total':statistics.mean(r[m] for r in per),'completed':sum(r[m+'_completed'] for r in per),'deducted':statistics.mean(r[m+'_deducted'] for r in per)} for m in models},
  'task_completion':sum(t['task_completed'] for t in trials),'planned_search_completion':sum(t['planned_search_completed'] for t in trials),
  'wall_limit_count':sum(t['stop_reason']=='wall_limit' for t in trials),
  'evaluation_limit_count':sum(t['stop_reason']=='evaluation_limit' for t in trials),
  'search':{k:{'mean':statistics.mean(t[k] for t in trials),'min':min(t[k] for t in trials),'max':max(t[k] for t in trials)} for k in
   ('evaluations','transitions','search_seconds','replay_seconds','per_candidate_seconds','outer_rounds','full_quotas','partial_quotas','fallback_steps','fallback_ratio','refresh_events','improvement')}}
# Holm over 4 primary
order=sorted(range(4),key=lambda i:all_primary[i]['p_boot']); rem=4
for i in order:
 all_primary[i]['p_holm']=min(1.0,all_primary[i]['p_boot']*rem); rem-=1
for i,x in enumerate(all_primary):
 # attach to scale's primary list by matching name/scale
 for scale in scales:
  if result[scale]['fields']==x['scale']:
   for y in result[scale]['primary']:
    if y['name']==x['name']:
     y['p_holm']=x['p_holm']
result['primary_all']=all_primary
(OUT/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
print(json.dumps({s:{'fields':result[s]['fields'],'ga':result[s]['ga_mean'],'controls':result[s]['controls_mean'],'primary':result[s]['primary'],'aux':result[s]['aux'],'task':result[s]['task_completion'],'planned':result[s]['planned_search_completion'],'wall':result[s]['wall_limit_count'],'search':result[s]['search']} for s in scales},indent=2))
