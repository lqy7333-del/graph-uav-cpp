"""Summarize 30 instances x single seed42 nested-GA independent comparison."""
import csv,json,statistics
from pathlib import Path
import numpy as np
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga/independent10')
OUT=BASE/'report_seed42'; OUT.mkdir(parents=True,exist_ok=True)
manifest=json.loads((BASE/'panel/manifest.json').read_text()); seeds=[int(x['seed']) for x in manifest['instances']]
assert len(seeds)==30
results=[]
for s in seeds:
 p=BASE/'runs'/f'instance_{s}_ga_42.json'; assert p.exists(),p; results.append(json.loads(p.read_text()))
controls=list(csv.DictReader((BASE/'controls/rows.csv').open())); cby={(int(r['instance_seed']),r['model']):r for r in controls}
trials=[]; progress=[]
for d in results:
 b=d['best']; q=d['inner_quotas']; src=b.get('fallback_sources',{}); seed=d['instance_seed']
 nfull=sum(1 for x in q if x['completed']); npart=sum(1 for x in q if not x['completed'])
 trial={'instance_seed':seed,'search_seed':42,'snapshot_sha256':d.get('instance_snapshot_sha256', d.get('snapshot_sha256')),
  'task_completed':bool(b['completed']),'stop_reason':d['termination'],'cost':b['cost'],
  'total_time':b['summary']['total_time'],'work':b['summary']['time_work_base'],
  'nonwork':b['summary']['time_nonwork_base'],'maneuver':b['summary']['time_maneuver_extra'],
  'service':b['summary']['time_service'],'distance':b['summary']['total_distance'],
  'visits':b['summary']['execution_count'],'resupplies':b['summary']['resupply_count'],
  'evaluations':d['evaluations'],'transitions':d['transitions'],'search_seconds':d['search_seconds'],
  'replay_seconds':d['replay_seconds'],'per_candidate_seconds':d['search_seconds']/d['evaluations'],
  'outer_generations_completed':d.get('outer_generations_completed'),
  'outer_candidates_attempted':d.get('outer_candidates_attempted'),
  'full_quotas':nfull,'partial_quotas':npart,
  'planned_search_completed':bool(d.get('outer_generations_completed')==6 and nfull==24 and npart==0),
  'fallback_steps':b.get('fallback_steps',0),'fallback_sources':src,
  'fallback_ratio':b.get('fallback_steps',0)/b['summary']['execution_count'],
  'refresh_events':sum(bool(h.get('improved')) for h in d['history']),
  'baseline_cost':d['baseline']['cost'],'improvement':d['baseline']['cost']-b['cost']}
 trials.append(trial)
 for h in d['history']:
  progress.append({'instance_seed':seed,'evaluation':h['evaluation'],'seconds':h['seconds'],'incumbent':h['best'],
   'completed':h['completed'],'cost':h['cost'],'label':h['label'],'generation':h.get('generation'),
   'candidate':h.get('candidate'),'improved':h.get('improved')})
models=['original_NN','ga_initializer','A10_u1060','A10_u1200']; per=[]
for t in trials:
 seed=t['instance_seed']; row={'instance_seed':seed,'ga':t['cost'],'task_completed':t['task_completed'],
  'stop_reason':t['stop_reason'],'planned_search_completed':t['planned_search_completed'],
  'evaluations':t['evaluations'],'outer_rounds':t['outer_generations_completed'],
  'full_quotas':t['full_quotas'],'partial_quotas':t['partial_quotas'],
  'fallback_steps':t['fallback_steps'],'fallback_sources':json.dumps(t['fallback_sources'],sort_keys=True),
  'per_candidate_seconds':t['per_candidate_seconds'],'ga_deducted':t['nonwork']+t['maneuver']+t['service']}
 for m in models:
  c=cby[(seed,m)]; row[m]=float(c['total_time']); row[m+'_completed']=str(c['completed']).lower()=='true'
  row[m+'_minus_ga']=float(c['total_time'])-t['cost']
  row[m+'_deducted']=float(c['time_nonwork_base'])+float(c['time_maneuver_extra'])+float(c['time_service'])
 per.append(row)
def boot(diffs):
 rng=np.random.default_rng(20261110); arr=np.asarray(diffs,float); vals=[]
 for _ in range(10000): vals.append(float(np.mean(rng.choice(arr,len(arr),replace=True))))
 return vals
def compare(name,diffs,relative_denom=None):
 arr=np.asarray(diffs,float); bs=np.asarray(boot(arr),float); lo,hi=np.percentile(bs,[2.5,97.5])
 p=(1+min(int(np.sum(bs<=0)),int(np.sum(bs>=0))))/(len(bs)+1); p=min(1.0,2*p)
 return {'name':name,'mean':float(arr.mean()),'median':float(np.median(arr)),'ci95':[float(lo),float(hi)],
  'wins':int(np.sum(arr>1e-6)),'ties':int(np.sum(np.abs(arr)<=1e-6)),'losses':int(np.sum(arr<-1e-6)),
  'p_boot':float(p),'relative_percent':float(arr.mean()/relative_denom*100) if relative_denom else None}
primary=[compare('NN - GA',[r['original_NN']-r['ga'] for r in per],statistics.mean(r['original_NN'] for r in per)),
         compare('GA - A10_u1060',[r['ga']-r['A10_u1060'] for r in per],statistics.mean(r['A10_u1060'] for r in per))]
order=sorted(range(len(primary)),key=lambda i:primary[i]['p_boot']); rem=2
for i in order:
 primary[i]['p_holm']=min(1.0,primary[i]['p_boot']*rem); rem-=1
aux=[compare('init - GA',[r['ga_initializer']-r['ga'] for r in per],statistics.mean(r['ga_initializer'] for r in per)),
     compare('GA - A10_u1200',[r['ga']-r['A10_u1200'] for r in per],statistics.mean(r['A10_u1200'] for r in per))]
def write_csv(path,rows):
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
write_csv(OUT/'ga_trials.csv',trials); write_csv(OUT/'per_instance.csv',per); write_csv(OUT/'progress.csv',progress)
summary={'panel':str(BASE/'panel'),'instances':30,'search_seed':42,'searches':30,
 'controls_mean':{m:{'total':statistics.mean(r[m] for r in per),'completed':sum(r[m+'_completed'] for r in per),
   'deducted':statistics.mean(r[m+'_deducted'] for r in per)} for m in models},
 'ga_mean':statistics.mean(r['ga'] for r in per),'ga_deducted_mean':statistics.mean(r['ga_deducted'] for r in per),
 'primary':primary,'aux':aux,'per_instance':per,'trials':trials,
 'task_completion':sum(t['task_completed'] for t in trials),'planned_search_completion':sum(t['planned_search_completed'] for t in trials),
 'wall_limit_count':sum(t['stop_reason']=='wall_limit' for t in trials),
 'evaluation_limit_count':sum(t['stop_reason']=='evaluation_limit' for t in trials),
 'search':{k:{'mean':statistics.mean(t[k] for t in trials),'min':min(t[k] for t in trials),'max':max(t[k] for t in trials)} for k in
   ('evaluations','transitions','search_seconds','replay_seconds','per_candidate_seconds','outer_generations_completed','full_quotas','partial_quotas','fallback_steps','fallback_ratio','refresh_events','improvement')}}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps({k:summary[k] for k in ('ga_mean','ga_deducted_mean','controls_mean','primary','aux','task_completion','planned_search_completion','wall_limit_count','evaluation_limit_count','search')},indent=2))
