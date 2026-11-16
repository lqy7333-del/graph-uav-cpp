"""M2 endpoint interaction, mechanism and common-budget analysis."""
import csv,json,pickle,statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_env.analysis.framework_mechanism import replay_result
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework/m2_payload'); OUT=BASE/'report'; OUT.mkdir(parents=True,exist_ok=True)
METHODS=('cc','divide_then_plan','plan_then_divide'); LAB={'cc':'CC','divide_then_plan':'D→P','plan_then_divide':'P→D'}
panel=json.loads((BASE/'panel/manifest.json').read_text()); seeds=[int(e['seed']) for e in panel['instances']]
rows=[]; histories={}
for f in sorted((BASE/'results').glob('instance_*.json')):
 d=json.loads(f.read_text()); s=d['summary']; r=d['result']; seed=s['seed']; cap=float(s['capacity']); method=s['method']
 env=pickle.loads((BASE/'panel'/f'instance_{seed}.pkl').read_bytes()); u=env.uavs[0]; u.max_payload=cap; u.remaining_payload=cap
 x=replay_result(env,r,method,{'id':f'instance_{seed}','fields':10}); vs=x['visits']; es=x['events']; ts=x['trips']
 rs=[e for e in es if e['type']=='resupply']; cov=sum(float(v['new_area']) for v in vs); ret_non=sum(float(e['nonwork']) for e in es)
 rows.append(dict(seed=seed,capacity=cap,map_id=f'instance_{seed}',method=method,
   cost=float(s['cost']),work=float(r['best']['summary']['time_work_base']),nonwork=float(r['best']['summary']['time_nonwork_base']),
   maneuver=float(r['best']['summary']['time_maneuver_extra']),service=float(r['best']['summary']['time_service']),
   total_minus_work=float(s['cost'])-float(r['best']['summary']['time_work_base']),
   visits=len(vs),departures=len(ts),resupplies=len(rs),coverage_per_trip=cov/len(ts) if ts else None,
   resupply_before_payload_ratio=statistics.mean(float(e['before_payload_ratio']) for e in rs) if rs else None,
   resupply_before_energy_ratio=statistics.mean(float(e['before_energy_ratio']) for e in rs) if rs else None,
   return_nonwork=ret_non,return_time=sum(float(e['total']) for e in es),
   nonreturn_nonwork=float(r['best']['summary']['time_nonwork_base'])-ret_non,
   return_distance=sum(float(e['return_distance']) for e in es),
   scan_segments_total=sum(float(v['scan_segments']) for v in vs),internal_connections_total=sum(float(v['scan_transit_segments']) for v in vs),
   evaluations=float(s['evaluations']),search_seconds=float(s['search_seconds']),full_rounds=sum(bool(x.get('completed_round')) for x in r.get('outer_records',[])),
   termination=s['termination'],completed=bool(s['completed']),failure=s['failure'],snapshot_sha256=s['snapshot_sha256'],
   replay_max_diff=x['replay']['max_ledger_diff']))
 histories[(seed,cap,method)]=r['history']
assert len(rows)==120 and all(r['completed'] and r['failure'] is None for r in rows)
with open(OUT/'m2_cells.csv','w',newline='') as fh:
 w=csv.DictWriter(fh,fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
by={(r['seed'],r['capacity'],r['method']):r for r in rows}
# endpoint d and interaction
inter=[]; endpoint=[]
for seed in seeds:
 for other in ('divide_then_plan','plan_then_divide'):
  d25=by[(seed,25.0,other)]['cost']-by[(seed,25.0,'cc')]['cost']
  d100=by[(seed,100.0,other)]['cost']-by[(seed,100.0,'cc')]['cost']
  inter.append({'seed':seed,'comparison':other,'d25':d25,'d100':d100,'interaction':d25-d100})
  endpoint += [{'seed':seed,'capacity':25.0,'comparison':other,'diff':d25},{'seed':seed,'capacity':100.0,'comparison':other,'diff':d100}]
# bootstrap
def boot(vals):
 rng=np.random.default_rng(20261117); arr=np.asarray(vals,float); out=[]
 for _ in range(10000): out.append(float(np.mean(rng.choice(arr,len(arr),replace=True))))
 return np.asarray(out)
report={'cells':120,'maps':20,'search_seconds':sum(r['search_seconds'] for r in rows),'completed':sum(r['completed'] for r in rows),'failures':sum(r['failure'] is not None for r in rows),
        'wall_counts':{str(c):{m:sum(r['termination']=='wall_limit' for r in rows if r['capacity']==c and r['method']==m) for m in METHODS} for c in (25.0,100.0)},
        'mean_eval':{str(c):{m:statistics.mean(r['evaluations'] for r in rows if r['capacity']==c and r['method']==m) for m in METHODS} for c in (25.0,100.0)},
        'endpoint_means':{},'interaction':{},'mechanism':{},'common_budget':{}}
for c in (25.0,100.0):
 report['endpoint_means'][str(c)]={}
 for m in METHODS:
  xs=[r for r in rows if r['capacity']==c and r['method']==m]
  report['endpoint_means'][str(c)][m]={k:statistics.mean(x[k] for x in xs if x[k] is not None) for k in ('cost','total_minus_work','work','nonwork','maneuver','resupplies','departures','coverage_per_trip','resupply_before_payload_ratio','return_nonwork','return_time','nonreturn_nonwork','maneuver','evaluations','search_seconds','full_rounds')}
for other in ('divide_then_plan','plan_then_divide'):
 xs=[x for x in inter if x['comparison']==other]; arr=np.array([x['interaction'] for x in xs]); bs=boot(arr); lo,hi=np.percentile(bs,[2.5,97.5]); p=(1+min(int((bs<=0).sum()),int((bs>=0).sum())))/(len(bs)+1); p=min(1.0,2*p)
 report['interaction'][other]={'n_maps':len(arr),'mean':float(arr.mean()),'median':float(np.median(arr)),'ci95':[float(lo),float(hi)],'p_boot':float(p),
   'positive':int((arr>1e-6).sum()),'negative':int((arr<-1e-6).sum()),'per_map':xs}
for c in (25.0,100.0):
 report['endpoint_means'][str(c)]['compare']={}
 for other in ('divide_then_plan','plan_then_divide'):
  arr=np.array([by[(s,c,other)]['cost']-by[(s,c,'cc')]['cost'] for s in seeds]); report['endpoint_means'][str(c)]['compare'][other]={'mean':float(arr.mean()),'median':float(np.median(arr)),'positive':int((arr>1e-6).sum()),'negative':int((arr<-1e-6).sum())}
body={}
# common candidate budget: min eval across the three methods per map/cap; best complete at/below cutoff
common=[]
for s in seeds:
 for c in (25.0,100.0):
  cut=min(by[(s,c,m)]['evaluations'] for m in METHODS)
  for m in METHODS:
   cost=None
   for h in histories[(s,c,m)]:
    if h.get('completed') and h.get('cost') is not None and h.get('evaluation',10**9)<=cut:
     cost=min(cost,h['cost']) if cost is not None else h['cost']
   common.append({'seed':s,'capacity':c,'method':m,'cutoff_eval':cut,'cost_at_common':cost})
for c in (25.0,100.0):
 report['common_budget'][str(c)]={}
 for other in ('divide_then_plan','plan_then_divide'):
  arr=[]
  for s in seeds:
   cc=next(x for x in common if x['seed']==s and x['capacity']==c and x['method']=='cc')
   o=next(x for x in common if x['seed']==s and x['capacity']==c and x['method']==other)
   if cc['cost_at_common'] is not None and o['cost_at_common'] is not None: arr.append(o['cost_at_common']-cc['cost_at_common'])
  report['common_budget'][str(c)][other]={'n':len(arr),'mean':float(np.mean(arr)) if arr else None,'median':float(np.median(arr)) if arr else None}
# Holm 2 interactions, monotone
order=sorted(report['interaction'],key=lambda k:report['interaction'][k]['p_boot']); rem=2; last=0
for k in order:
 adj=min(1.0,report['interaction'][k]['p_boot']*rem); last=max(last,adj); report['interaction'][k]['p_holm']=last; rem-=1
(OUT/'m2_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
# plots
fig,ax=plt.subplots(figsize=(7.2,4.5))
for other,col in (('divide_then_plan','#1f4e9c'),('plan_then_divide','#2e8b57')):
 xs=[25,100]; ys=[report['endpoint_means'][str(c)]['compare'][other]['mean'] for c in (25.0,100.0)]
 ax.plot(xs,ys,'o-',color=col,lw=2,label=LAB[other]+'−CC')
 for s in seeds:
  ax.plot(xs,[by[(s,25.0,other)]['cost']-by[(s,25.0,'cc')]['cost'],by[(s,100.0,other)]['cost']-by[(s,100.0,'cc')]['cost']],color=col,alpha=.15,lw=.7)
ax.axhline(0,color='k',lw=.8); ax.set_xticks([25,100]); ax.set_xticklabels(['25 (0.5×)','100 (2×)']); ax.set_xlabel('payload capacity'); ax.set_ylabel('other − CC total (s)'); ax.set_title('M2 endpoint interaction (positive=CC faster)'); ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT/'interaction.png',dpi=190,bbox_inches='tight'); plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7))
for ax,(metric,label) in zip(axes.flat,(('cost','mean total cost'),('resupplies','resupplies'),('return_nonwork','return nonwork'),('maneuver','maneuver'))):
 for m,col in (('cc','#555'),('divide_then_plan','#1f4e9c'),('plan_then_divide','#2e8b57')):
  vals=[statistics.mean(r[metric] for r in rows if r['capacity']==c and r['method']==m) for c in (25.0,100.0)]
  ax.plot([25,100],vals,'o-',color=col,label=LAB[m])
 ax.set_xticks([25,100]); ax.set_xticklabels(['25','100']); ax.set_title(label); ax.grid(alpha=.25)
axes.flat[0].legend(); fig.tight_layout(); fig.savefig(OUT/'mechanism.png',dpi=190,bbox_inches='tight'); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,4.5))
for other,col in (('divide_then_plan','#1f4e9c'),('plan_then_divide','#2e8b57')):
 ys=[report['common_budget'][str(c)][other]['mean'] for c in (25.0,100.0)]; ax.plot([25,100],ys,'o-',color=col,label=LAB[other]+'−CC common budget')
ax.axhline(0,color='k',lw=.8); ax.set_xticks([25,100]); ax.set_xticklabels(['25','100']); ax.set_xlabel('payload capacity'); ax.set_ylabel('common-candidate-budget difference (s)'); ax.set_title('Auxiliary common-budget view'); ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT/'common_budget.png',dpi=190,bbox_inches='tight'); plt.close(fig)
print(json.dumps(report,indent=2)[:8000])
