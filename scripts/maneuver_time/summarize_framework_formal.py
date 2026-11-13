"""Summarize framework formal 10/20 shared-panel results."""
import csv,json,statistics
from pathlib import Path
import numpy as np
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework'); OUT=BASE/'formal_report'; OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for scale in ('formal10_run','formal20_run'):
 d=json.loads((BASE/scale/'rows.json').read_text()); rows += d
# index result extras by id/method
extras={}
for scale in ('formal10_run','formal20_run'):
 for f in (BASE/scale).glob('*.json'):
  if f.name in ('rows.json','summary.json','run_contract.json','progress.json','acceptance_receipt.json','search_audit.json','failure.json'): continue
  try: d=json.loads(f.read_text())
  except: continue
  if 'identity' not in d or 'result' not in d: continue
  extras[(d['identity']['id'],d['identity']['method'])]=d['result']
audits={}
for scale in ('formal10_run','formal20_run'):
 for a in json.loads((BASE/scale/'search_audit.json').read_text()): audits[(a['id'],a['method'])]=a
methods=['cc','divide_then_plan','plan_then_divide']
by={}
for r in rows: by.setdefault((r['fields'],r['id']),{})[r['method']]=r
scales={}
for fields in (10,20):
 items=[v for (f,i),v in by.items() if f==fields]
 assert len(items)==30
 # complete only if all three complete; retain failures
 comp={m:sum(bool(v.get(m) and v[m]['completed']) for v in items) for m in methods}
 prim=[]; aux=[]
 def boot(d):
  rng=np.random.default_rng(20261113); arr=np.asarray(d,float); vals=[]
  for _ in range(10000): vals.append(float(np.mean(rng.choice(arr,len(arr),replace=True))))
  return np.asarray(vals)
 def cmp(name,d,denom):
  arr=np.asarray(d,float); bs=boot(d); lo,hi=np.percentile(bs,[2.5,97.5]); p=min(1.0,2*(1+min(int(np.sum(bs<=0)),int(np.sum(bs>=0))))/(len(bs)+1))
  return {'name':name,'mean':float(arr.mean()),'median':float(np.median(arr)),'ci95':[float(lo),float(hi)],'wins':int(np.sum(arr>1e-6)),'ties':int(np.sum(np.abs(arr)<=1e-6)),'losses':int(np.sum(arr<-1e-6)),'p_boot':float(p),'relative_percent':float(arr.mean()/denom*100)}
 valid=[v for v in items if all(v.get(m) and v[m]['completed'] for m in methods)]
 ccm=statistics.mean(v['cc']['total_time'] for v in valid)
 prim += [cmp('divide_then_plan - cc',[v['divide_then_plan']['total_time']-v['cc']['total_time'] for v in valid],ccm),
          cmp('plan_then_divide - cc',[v['plan_then_divide']['total_time']-v['cc']['total_time'] for v in valid],ccm)]
 for m in methods:
  cs=[v[m] for v in valid]
  aux.append({'method':m,'completed':sum(bool(v[m]['completed']) for v in items),'n_pairs':len(valid),
   'mean_total_time':statistics.mean(x['total_time'] for x in cs),
   'mean_work':statistics.mean(x['work'] for x in cs),'mean_nonwork':statistics.mean(x['nonwork'] for x in cs),
   'mean_maneuver':statistics.mean(x['maneuver'] for x in cs),'mean_service':statistics.mean(x['service'] for x in cs),
   'mean_deducted':statistics.mean(x['excess_time'] for x in cs),
   'mean_search_seconds':statistics.mean(x['search_seconds'] for x in cs),
   'mean_replay_seconds':statistics.mean(extras[(x['id'],m)].get('replay_seconds',0) for x in cs),
   'mean_evaluations':statistics.mean(x['evaluations'] for x in cs),
   'wall_limit':sum(x['termination']=='wall_limit' for x in cs),
   'full_rounds_mean':statistics.mean(x['full_outer_rounds'] for x in cs),
   'full_rounds_min':min(x['full_outer_rounds'] for x in cs),
   'full_rounds_max':max(x['full_outer_rounds'] for x in cs),
   'transitions_mean':statistics.mean(extras[(x['id'],m)].get('transitions',0) for x in cs),
   'coverage_min':min(extras[(x['id'],m)]['best']['summary']['coverage_ratio'] for x in cs),
   'visits_mean':statistics.mean(extras[(x['id'],m)]['best']['summary']['execution_count'] for x in cs),
   'resupplies_mean':statistics.mean(extras[(x['id'],m)]['best']['summary']['resupply_count'] for x in cs),
   'distance_mean':statistics.mean(extras[(x['id'],m)]['best']['summary']['total_distance'] for x in cs),
   'fallback_mean':statistics.mean(extras[(x['id'],m)]['best'].get('fallback_steps',0) for x in cs)})
 # audit/cache summaries
 p2d=[extras[(v['plan_then_divide']['id'],'plan_then_divide')] for v in valid]
 d2p=[extras[(v['divide_then_plan']['id'],'divide_then_plan')] for v in valid]
 audit_p=[audits[(v['plan_then_divide']['id'],'plan_then_divide')] for v in valid]
 audit_d=[audits[(v['divide_then_plan']['id'],'divide_then_plan')] for v in valid]
 special={'p2d_decode_cache_hits_mean':statistics.mean(x.get('decode_cache_hits',0) for x in p2d),
   'p2d_decode_cache_misses_mean':statistics.mean(x.get('decode_cache_misses',0) for x in p2d),
   'p2d_decode_saved_seconds_mean':statistics.mean(x.get('decode_cache_saved_seconds',0) for x in p2d),
   'p2d_repair_probes_mean':statistics.mean(x.get('repair_probes',0) for x in p2d),
   'p2d_repair_added_mean':statistics.mean(x.get('best',{}).get('repair_added_visits',0) for x in p2d),
   'p2d_area_error_abs_max':max(max(abs(e) for e in x.get('best',{}).get('division_audit',[])[0].keys() if False) for x in []) if False else None,
   'p2d_distinct_chromosomes_mean':statistics.mean(a.get('distinct_chromosomes') or 0 for a in audit_p),
   'p2d_distinct_partitions_mean':statistics.mean(a.get('distinct_partitions') or 0 for a in audit_p),
   'p2d_max_added_visits':max((a.get('maximum_added_visits') or 0) for a in audit_p),
   'd2p_preprocess_seconds_mean':statistics.mean(x.get('preprocessing_seconds') or 0 for x in d2p),
   'd2p_probes_mean':statistics.mean(x.get('preprocessing_probes') or 0 for x in d2p),
   'd2p_distinct_chromosomes_mean':statistics.mean(a.get('distinct_chromosomes') or 0 for a in audit_d),
   'd2p_distinct_costs_mean':statistics.mean(a.get('distinct_complete_costs') or 0 for a in audit_d)}
 # max area error from p2d best traces
 errs=[]
 for x in p2d:
  for e in x.get('best',{}).get('division_audit',[]): errs.append(abs(e.get('area_error',0)))
 special['p2d_area_error_abs_max']=max(errs) if errs else None
 special['p2d_area_error_abs_mean']=statistics.mean(errs) if errs else None
 scales[str(fields)]={'n_instances':30,'complete_counts':comp,'primary':prim,'aux':aux,'valid_pairs':len(valid),'special':special}
# Holm 4 primary
allp=[]
for f in ('10','20'):
 for p in scales[f]['primary']: allp.append((f,p))
order=sorted(range(4),key=lambda i:allp[i][1]['p_boot']); rem=4
last_adj=0.0
for i in order:
 adj=min(1.0,allp[i][1]['p_boot']*rem)
 last_adj=max(last_adj,adj)
 allp[i][1]['p_holm']=last_adj
 rem-=1
(OUT/'summary.json').write_text(json.dumps(scales,indent=2,ensure_ascii=False))
# per-pair CSV
with (OUT/'paired.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['fields','id','cc','divide_then_plan','plan_then_divide','d2p_minus_cc','p2d_minus_cc','d2p_completed','p2d_completed'])
 for (fields,i),v in sorted(by.items()):
  if not all(m in v for m in methods): continue
  cc=v['cc']['total_time'] if v['cc']['completed'] else None; d=v['divide_then_plan']['total_time'] if v['divide_then_plan']['completed'] else None; p=v['plan_then_divide']['total_time'] if v['plan_then_divide']['completed'] else None
  w.writerow([fields,i,cc,d,p,None if cc is None or d is None else d-cc,None if cc is None or p is None else p-cc,v['divide_then_plan']['completed'],v['plan_then_divide']['completed']])
print(json.dumps(scales,indent=2))
