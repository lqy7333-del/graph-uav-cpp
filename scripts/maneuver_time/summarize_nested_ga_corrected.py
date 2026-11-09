"""Summarize the corrected 4x6 nested-GA pilot on the same five snapshots."""
import csv, json, statistics
from pathlib import Path
EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments')
PILOT=EXP/'nested_ga/pilot10_corrected_s1400000'; CTRL=EXP/'nested_ga/controls10_s1400000'; OUT=EXP/'nested_ga/report10_corrected_s1400000'
OUT.mkdir(parents=True,exist_ok=True)
trials=[]; curves=[]; events=[]
for path in sorted(PILOT.glob('instance_*_ga_*.json')):
 d=json.loads(path.read_text()); b=d['best']; q=d['inner_quotas']
 full=sum(1 for x in q if x['completed']); partial=sum(1 for x in q if not x['completed'])
 imps=[h for h in d['history'] if h.get('improved')]
 trial={'instance_seed':d['instance_seed'],'search_seed':d['config']['seed'],'snapshot_sha256':d['instance_snapshot_sha256'],
  'best_cost':b['cost'],'baseline_cost':d['baseline']['cost'],'improvement':d['baseline']['cost']-b['cost'],
  'completed':bool(b['completed']),'termination':d['termination'],'evaluations':d['evaluations'],'transitions':d['transitions'],
  'search_seconds':d['search_seconds'],'replay_seconds':d['replay_seconds'],'total_seconds':d['search_seconds']+d['replay_seconds'],
  'outer_generations_completed':d['outer_generations_completed'],'outer_generations_attempted':d['outer_generations_attempted'],
  'outer_candidates_attempted':d['outer_candidates_attempted'],'outer_candidates_completed':d['outer_candidates_completed'],
  'full_quotas':full,'partial_quotas':partial,'quotas':len(q),'fallback_steps':b['fallback_steps'],
  'fallback_sources':b['fallback_sources'],'fallback_ratio':b['fallback_steps']/b['summary']['execution_count'],
  'skipped_slots':b['skipped_slots'],'execution_steps':b['summary']['execution_count'],
  'total_time':b['summary']['total_time'],'work':b['summary']['time_work_base'],'nonwork':b['summary']['time_nonwork_base'],
  'maneuver':b['summary']['time_maneuver_extra'],'service':b['summary']['time_service'],
  'distance':b['summary']['total_distance'],'visits':b['summary']['execution_count'],'resupplies':b['summary']['resupply_count']}
 trials.append(trial)
 for h in d['history']:
  curves.append({'instance_seed':d['instance_seed'],'evaluation':h['evaluation'],'seconds':h['seconds'],
                 'incumbent':h['best'],'completed':h['completed'],'cost':h['cost'],'label':h['label'],
                 'generation':h['generation'],'candidate':h['candidate'],'improved':h['improved']})
 for h in imps:
  events.append({'instance_seed':d['instance_seed'],'evaluation':h['evaluation'],'seconds':h['seconds'],
                 'generation':h['generation'],'candidate':h['candidate'],'label':h['label'],
                 'source':h['best_source'],'best':h['best']})
controls=list(csv.DictReader((CTRL/'rows.csv').open()))
models=['original_NN','ga_initializer','A10_u1060','A10_u1200']
per=[]
for seed in sorted(t['instance_seed'] for t in trials):
 tr=[t for t in trials if t['instance_seed']==seed][0]
 row={'instance_seed':seed,'ga':tr['best_cost'],'improvement':tr['improvement'],
      'fallback_sources':json.dumps(tr['fallback_sources'],sort_keys=True),
      'nonwork_maneuver_service':tr['nonwork']+tr['maneuver']+tr['service']}
 for m in models:
  rr=[r for r in controls if int(r['instance_seed'])==seed and r['model']==m][0]
  row[m]=float(rr['total_time']); row[m+'_minus_ga']=float(rr['total_time'])-tr['best_cost']
  row[m+'_nonwork_maneuver_service']=float(rr['time_nonwork_base'])+float(rr['time_maneuver_extra'])+float(rr['time_service'])
 per.append(row)
keys=list(trials[0])
with (OUT/'corrected_trials.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(trials)
pkeys=['instance_seed','ga','improvement','fallback_sources','nonwork_maneuver_service']+[m+s for m in models for s in ('','_minus_ga','_nonwork_maneuver_service')]
with (OUT/'per_instance.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=pkeys); w.writeheader(); w.writerows(per)
with (OUT/'progress.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['instance_seed','evaluation','seconds','incumbent','completed','cost','label','generation','candidate','improved']); w.writeheader(); w.writerows(curves)
summary={'pilot':str(PILOT),'controls':str(CTRL),'instances':5,'search_seeds':[42],'searches':len(trials),
 'trials':trials,'per_instance':per,'improvement_events':events,
 'controls_mean':{m:{'total':statistics.mean(r[m] for r in per),
   'deducted':statistics.mean(r[m+'_nonwork_maneuver_service'] for r in per)} for m in models},
 'ga_mean':statistics.mean(t['best_cost'] for t in trials),
 'ga_deducted_mean':statistics.mean(t['nonwork']+t['maneuver']+t['service'] for t in trials),
 'search':{k:{'mean':statistics.mean(t[k] for t in trials),'min':min(t[k] for t in trials),'max':max(t[k] for t in trials)} for k in
   ('evaluations','transitions','search_seconds','replay_seconds','total_seconds','outer_generations_completed','outer_candidates_attempted','full_quotas','partial_quotas','fallback_steps','fallback_ratio','skipped_slots','improvement')}}
for m in models:
 ds=[r[m+'_minus_ga'] for r in per]
 summary.setdefault('control_minus_ga',{})[m]={'mean':statistics.mean(ds),'min':min(ds),'max':max(ds),'ga_faster_instances':sum(x>1e-6 for x in ds)}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps({'ga_mean':summary['ga_mean'],'ga_deducted':summary['ga_deducted_mean'],'controls_mean':summary['controls_mean'],'control_minus_ga':summary['control_minus_ga'],'search':summary['search']},indent=2))
