"""M0 paired mechanism statistics and exploratory feature associations."""
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework'); M=BASE/'mechanism_m0'; OUT=M
METHODS=('cc','divide_then_plan','plan_then_divide'); PAIRS=(('divide_then_plan','cc'),('plan_then_divide','cc'))

def read(name): return list(csv.DictReader(open(M/name)))
def f(x):
    if x in (None,'','None','null'): return None
    try: return float(x)
    except: return None
rows=[]; extras={}
for scale,run in ((10,'formal10_run'),(20,'formal20_run')):
    rows += json.loads((BASE/run/'rows.json').read_text())
    for p in (BASE/run).glob('*.json'):
        try: d=json.loads(p.read_text())
        except: continue
        if 'identity' in d and 'result' in d: extras[(d['identity']['id'],d['identity']['method'])]=d['result']
audits={}
for scale,run in ((10,'formal10_run'),(20,'formal20_run')):
    for a in json.loads((BASE/run/'search_audit.json').read_text()): audits[(a['id'],a['method'])]=a
visits=read('visits.csv'); trips=read('trips.csv'); events=read('events.csv'); features={r['instance_id']:r for r in read('features.csv')}
fieldfeat=read('field_features.csv'); fagg={}
for ff in fieldfeat:
 key=(ff['scale'],ff['instance_id']); fagg.setdefault(key,[]).append(ff)
for (scale,i),rowsf in fagg.items():
 def vals(k):
  return [float(x[k]) for x in rowsf if x.get(k) not in (None,'','None')]
 a=vals('area'); asp=vals('aspect_ratio'); dep=vals('normalized_depot_distance'); near=vals('nearest_normalized'); aq=vals('area_over_qmax')
 f=features[i]
 f.update({'field_count':len(rowsf),'aspect_ratio_mean':statistics.mean(asp) if asp else None,'aspect_ratio_max':max(asp) if asp else None,
  'depot_normalized_mean':statistics.mean(dep) if dep else None,'depot_normalized_max':max(dep) if dep else None,
  'nearest_normalized_mean':statistics.mean(near) if near else None,'nearest_normalized_min':min(near) if near else None,
  'area_over_qmax_mean':statistics.mean(aq) if aq else None,'area_over_qmax_max':max(aq) if aq else None})
# per-instance/method metrics
by=defaultdict(dict)
for r in rows: by[(int(r['fields']),r['id'])][r['method']]=r
vm=defaultdict(list)
for v in visits: vm[(int(v['scale']),v['instance_id'],v['method'])].append(v)
tm=defaultdict(list)
for t in trips: tm[(int(t['scale']),t['instance_id'],t['method'])].append(t)
em=defaultdict(list)
for e in events: em[(int(e['scale']),e['instance_id'],e['method'])].append(e)
metrics={}
for key,methods in by.items():
    scale,i=key; metrics[key]={}
    for m in METHODS:
        r=methods.get(m); vs=vm.get((scale,i,m),[]); ts=tm.get((scale,i,m),[]); es=em.get((scale,i,m),[])
        if r is None or not vs: metrics[key][m]=None; continue
        total=float(r['total_time']); work=float(r['work']); non=float(r['nonwork']); man=float(r['maneuver']); svc=float(r['service'])
        rs=[e for e in es if e['type']=='resupply']
        ret=[e for e in es if e['type']=='terminal_return']
        cov=sum(float(v['new_area']) for v in vs); assigned=sum(float(v['assigned_area']) for v in vs)
        comps=[float(v['region_components']) for v in vs if v['region_components'] not in ('','None')]
        metrics[key][m]=dict(total=total,work=work,nonwork=non,maneuver=man,service=svc,total_minus_work=total-work,
            visits=len(vs),trips=len(ts),resupplies=len(rs),terminal_returns=len(ret),
            visits_per_trip=len(vs)/len(ts) if ts else None,coverage_per_trip=cov/len(ts) if ts else None,
            coverage=cov,assigned=assigned,overlap=assigned-cov,overlap_ratio=(assigned-cov)/assigned if assigned else None,
            region_components_mean=statistics.mean(comps) if comps else None,
            multi_component_visit_frac=sum(c>1 for c in comps)/len(comps) if comps else None,
            resupply_before_energy=statistics.mean(float(e['before_energy']) for e in rs) if rs else None,
            resupply_before_energy_ratio=statistics.mean(float(e['before_energy_ratio']) for e in rs) if rs else None,
            resupply_before_payload=statistics.mean(float(e['before_payload']) for e in rs) if rs else None,
            resupply_before_payload_ratio=statistics.mean(float(e['before_payload_ratio']) for e in rs) if rs else None,
            return_distance=sum(float(e['return_distance']) for e in es),
            return_time=sum(float(e['total']) for e in es),
            coverage_distance=sum(float(v['coverage_distance'] or 0) for v in vs),
            transfer_distance=sum(float(v['transfer_distance'] or 0) for v in vs),
            evaluation=float(r['evaluations']),search_seconds=float(r['search_seconds']),
            full_rounds=float(r['full_outer_rounds']),wall=1.0 if r['termination']=='wall_limit' else 0.0,
            repair_added=float((extras.get((i,m)) or {}).get('best',{}).get('repair_added_visits',0) or 0) if m=='plan_then_divide' else None,
            d2p_probes=float((extras.get((i,m)) or {}).get('preprocessing_probes',0) or 0) if m=='divide_then_plan' else None)
            # exact component closure available from replay
# write per-instance/average
wide=[]
for (scale,i),ms in sorted(metrics.items()):
    if any(ms.get(m) is None for m in METHODS): continue
    rec={'scale':scale,'instance_id':i}
    for m in METHODS:
        for k,v in ms[m].items(): rec[f'{m}_{k}']=v
    for other,_ in PAIRS:
        for k in ms[other]:
            a=ms[other][k]; b=ms['cc'][k]
            rec[f'{other}_minus_cc_{k}']=None if a is None or b is None else a-b
    for k,v in features[i].items():
        if k not in ('instance_id','scale'): rec[f'feature_{k}']=v
    wide.append(rec)
with open(OUT/'mechanism_instance.csv','w',newline='') as fh:
    keys=sorted({k for r in wide for k in r}); w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(wide)
with open(OUT/'features_enriched.csv','w',newline='') as fh:
    keys=sorted({k for r in features.values() for k in r}); w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(features.values())

def boot(arr):
    rng=np.random.default_rng(20261114); arr=np.asarray(arr,float); vals=[]
    for _ in range(10000): vals.append(float(np.mean(rng.choice(arr,len(arr),replace=True))))
    return np.asarray(vals)
def effect(scale,other,k):
    arr=np.array([r[f'{other}_minus_cc_{k}'] for r in wide if r['scale']==scale and r[f'{other}_minus_cc_{k}'] is not None],float)
    if not len(arr): return None
    bs=boot(arr); lo,hi=np.percentile(bs,[2.5,97.5]); cc=np.mean([r[f'cc_{k}'] for r in wide if r['scale']==scale and r[f'cc_{k}'] is not None])
    # p omitted for the mechanism family; main cost tests already Holm-corrected in formal report
    return dict(n=len(arr),mean=float(arr.mean()),median=float(np.median(arr)),ci95=[float(lo),float(hi)],
                relative_percent=float(arr.mean()/cc*100) if cc else None,
                positive=int((arr>1e-6).sum()),ties=int((abs(arr)<=1e-6).sum()),negative=int((arr<-1e-6).sum()))
metric_names=['total','total_minus_work','work','nonwork','maneuver','service','visits','trips','resupplies','terminal_returns',
 'visits_per_trip','coverage_per_trip','coverage','assigned','overlap','overlap_ratio','region_components_mean',
 'multi_component_visit_frac','resupply_before_energy','resupply_before_energy_ratio','resupply_before_payload',
 'resupply_before_payload_ratio','return_distance','return_time','coverage_distance','transfer_distance','evaluation','search_seconds',
 'full_rounds','wall','repair_added','d2p_probes']
summary={}
for scale in (10,20):
    summary[str(scale)]={}
    for other,_ in PAIRS:
        summary[str(scale)][f'{other}_minus_cc']={k:effect(scale,other,k) for k in metric_names}
# correlations features vs diffs (exploratory; no p-value/selection)
feature_names=['total_area','field_area_mean','field_area_std','field_area_cv','max_area','min_area','qmax_area',
 'payload_per_area','field_count','aspect_ratio_mean','aspect_ratio_max','depot_normalized_mean','depot_normalized_max',
 'nearest_normalized_mean','nearest_normalized_min','area_over_qmax_mean','area_over_qmax_max']
corr=[]
for scale in (10,20):
    for other,_ in PAIRS:
        for k in ('total','total_minus_work','work','nonwork','maneuver','visits','trips','resupplies','coverage_per_trip','return_distance'):
            for name in feature_names:
                x=[]; y=[]
                for r in wide:
                    if r['scale']!=scale: continue
                    yv=r[f'{other}_minus_cc_{k}']; xv=r.get('feature_'+name)
                    if yv is None or xv is None: continue
                    x.append(float(xv)); y.append(float(yv))
                if len(x)>=3 and np.std(x)>0 and np.std(y)>0:
                    corr.append({'scale':scale,'comparison':f'{other}_minus_cc','metric':k,'feature':name,
                                 'pearson':float(np.corrcoef(x,y)[0,1]),'n':len(x)})
# cases at 25/50/75 quantiles for each scale/comparison by total difference
cases=[]
for scale in (10,20):
    for other,_ in PAIRS:
        vals=sorted([(r[f'{other}_minus_cc_total'],r['instance_id']) for r in wide if r['scale']==scale and r[f'{other}_minus_cc_total'] is not None])
        if not vals: continue
        for q in (.25,.5,.75):
            target=np.quantile([v for v,_ in vals],q); chosen=min(vals,key=lambda z:(abs(z[0]-target),z[1]))
            cases.append({'scale':scale,'comparison':f'{other}_minus_cc','quantile':q,'difference':chosen[0],'instance_id':chosen[1]})
# closure validation from replay per-row visits/trips not here; report via replay csv separately
summary['correlations']=corr; summary['cases']=cases
# closure check by instance/method from visit/event/trip ledgers
closure=[]
for (scale,i,m),vs in vm.items():
    if not vs: continue
    ev=em.get((scale,i,m),[]); ts=tm.get((scale,i,m),[])
    vsum=sum(float(v['total']) for v in vs); esum=sum(float(e['total']) for e in ev); tsum=sum(float(t['total']) for t in ts)
    row=next((r for r in rows if int(r['fields'])==int(scale) and r['id']==i and r['method']==m),None)
    if row:
        summary_t=float(row['total_time']); closure.append({'scale':scale,'instance_id':i,'method':m,'visit_event_minus_total':vsum+esum-summary_t,'trip_minus_total':tsum-summary_t})
summary['max_abs_visit_event_closure_error']=max(abs(x['visit_event_minus_total']) for x in closure)
summary['max_abs_trip_closure_error']=max(abs(x['trip_minus_total']) for x in closure)
# special repair/cache/probe values
special={}
for scale in (10,20):
    p2=[x for x in wide if x['scale']==scale]
    special[str(scale)]={'p2d_repair_added_mean':statistics.mean(float(x['plan_then_divide_repair_added']) for x in p2 if x['plan_then_divide_repair_added'] not in ('','None')),
      'd2p_probe_mean':statistics.mean(float(x['divide_then_plan_d2p_probes']) for x in p2 if x['divide_then_plan_d2p_probes'] not in ('','None')),
      'max_region_components':max(float(x[f'{m}_region_components_mean']) for x in p2 for m in METHODS),
      'max_overlap_ratio':max(float(x[f'{m}_overlap_ratio']) for x in p2 for m in METHODS)}
# P2D repair/diversity/cache and D2P fixed-partition evidence
import hashlib
for scale in (10,20):
    p2=[]; d2=[]
    for (i,m),res in extras.items():
        if not i.startswith('instance_1500' if scale==10 else 'instance_1800'): continue
        if m=='plan_then_divide': p2.append((i,res))
        if m=='divide_then_plan': d2.append((i,res))
    errs=[]
    for i,r in p2:
        for e in r.get('best',{}).get('division_audit',[]): errs.append(abs(e.get('area_error',0)))
    p2aud=[audits.get((i,'plan_then_divide'),{}) for i,_ in p2]
    special[str(scale)].update({
      'p2d_area_error_abs_mean':statistics.mean(errs) if errs else None,
      'p2d_area_error_abs_max':max(errs) if errs else None,
      'p2d_repair_added_max':max((r.get('best',{}).get('repair_added_visits',0) or 0) for _,r in p2),
      'p2d_cache_hits_mean':statistics.mean((r.get('decode_cache_hits') or 0) for _,r in p2),
      'p2d_cache_misses_mean':statistics.mean((r.get('decode_cache_misses') or 0) for _,r in p2),
      'p2d_cache_saved_seconds_mean':statistics.mean((r.get('decode_cache_saved_seconds') or 0) for _,r in p2),
      'p2d_distinct_chromosomes_mean':statistics.mean((a.get('distinct_chromosomes') or 0) for a in p2aud),
      'p2d_distinct_partitions_mean':statistics.mean((a.get('distinct_partitions') or 0) for a in p2aud),
      'd2p_preprocessing_seconds_mean':statistics.mean((r.get('preprocessing_seconds') or 0) for _,r in d2),
      'd2p_partition_hash_unique':len({hashlib.sha256(json.dumps(r.get('tasks'),sort_keys=True).encode()).hexdigest() for _,r in d2}),
      'd2p_tasks_mean':statistics.mean(len(r.get('tasks',[])) for _,r in d2)})
summary['special']=special
summary['closure_rows']=closure
(OUT/'mechanism_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
print(json.dumps({'instances':len(wide),'summary_keys':list(summary),'cases':len(cases)},indent=2))
