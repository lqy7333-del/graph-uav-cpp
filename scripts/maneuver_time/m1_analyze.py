"""M1 payload response: replay ledgers, descriptive interaction and mechanism plots."""
import csv,json,pickle,hashlib,statistics
from copy import deepcopy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_env.analysis.framework_mechanism import replay_result
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework/m1_payload'); OUT=BASE/'report'; OUT.mkdir(parents=True,exist_ok=True)
METHODS=('cc','divide_then_plan','plan_then_divide'); LAB={'cc':'CC','divide_then_plan':'D→P','plan_then_divide':'P→D'}
manifest=json.loads((BASE/'panel/manifest.json').read_text()); snaps={int(e['seed']):e for e in manifest['instances']}
rows=[]; ledgers=[]
for f in sorted((BASE/'results').glob('instance_*.json')):
 d=json.loads(f.read_text()); s=d['summary']; r=d['result']; seed=s['seed']; fac=float(s['payload_factor']); method=s['method']
 env=pickle.loads((BASE/'panel'/f'instance_{seed}.pkl').read_bytes()); u=env.uavs[0]; cap=float(u.max_payload)*fac; u.max_payload=cap; u.remaining_payload=cap
 x=replay_result(env,r,method,{'id':f'instance_{seed}','fields':10})
 vs=x['visits']; es=x['events']; ts=x['trips']; visits=len(vs); trips=len(ts); rs=[e for e in es if e['type']=='resupply']; terminal=[e for e in es if e['type']=='terminal_return']
 cov=sum(float(v['new_area']) for v in vs); ret_non=sum(float(e['nonwork']) for e in es); ret_time=sum(float(e['total']) for e in es)
 row=dict(seed=seed,map_id=f'instance_{seed}',factor=fac,capacity=cap,method=method,
  cost=float(s['cost']),work=float(r['best']['summary']['time_work_base']),nonwork=float(r['best']['summary']['time_nonwork_base']),
  maneuver=float(r['best']['summary']['time_maneuver_extra']),service=float(r['best']['summary']['time_service']),
  total_minus_work=float(s['cost'])-float(r['best']['summary']['time_work_base']),
  visits=visits,departures=trips,resupplies=len(rs),terminal_returns=len(terminal),
  coverage_per_trip=cov/trips if trips else None,
  resupply_before_payload_ratio=statistics.mean(float(e['before_payload_ratio']) for e in rs) if rs else None,
  resupply_before_energy_ratio=statistics.mean(float(e['before_energy_ratio']) for e in rs) if rs else None,
  return_nonwork=ret_non,return_time=ret_time,nonreturn_nonwork=float(r['best']['summary']['time_nonwork_base'])-ret_non,
  return_distance=sum(float(e['return_distance']) for e in es),
  scan_segments_total=sum(float(v['scan_segments']) for v in vs),internal_connections_total=sum(float(v['scan_transit_segments']) for v in vs),
  evaluations=float(s['evaluations']),search_seconds=float(s['search_seconds']),termination=s['termination'],
  completed=bool(s['completed']),replay_max_diff=x['replay']['max_ledger_diff'],failure=s['failure'],snapshot_sha256=s['snapshot_sha256'])
 rows.append(row); ledgers.append((row,x))
# per-cell checks
assert len(rows)==27, len(rows)
assert all(r['completed'] and r['failure'] is None for r in rows)
# write instance CSV
keys=sorted({k for r in rows for k in r})
with open(OUT/'payload_cells.csv','w',newline='') as fh:
 w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(rows)
# paired differences by map/factor
by={(r['seed'],r['factor'],r['method']):r for r in rows}
effects=[]
for seed in sorted({r['seed'] for r in rows}):
 for fac in (.5,1.0,2.0):
  cc=by[(seed,fac,'cc')]
  for other in ('divide_then_plan','plan_then_divide'):
   o=by[(seed,fac,other)]
   rec={'seed':seed,'factor':fac,'comparison':other+'_minus_cc','cost_diff':o['cost']-cc['cost'],
        'total_minus_work_diff':o['total_minus_work']-cc['total_minus_work'],
        'resupply_diff':o['resupplies']-cc['resupplies'],'departure_diff':o['departures']-cc['departures'],
        'payload_residual_diff':(o['resupply_before_payload_ratio'] or 0)-(cc['resupply_before_payload_ratio'] or 0),
        'return_nonwork_diff':o['return_nonwork']-cc['return_nonwork'],'return_time_diff':o['return_time']-cc['return_time'],
        'nonreturn_nonwork_diff':o['nonreturn_nonwork']-cc['nonreturn_nonwork'],'maneuver_diff':o['maneuver']-cc['maneuver'],
        'scan_segments_diff':o['scan_segments_total']-cc['scan_segments_total']}
   effects.append(rec)
with open(OUT/'payload_effects.csv','w',newline='') as fh:
 w=csv.DictWriter(fh,fieldnames=list(effects[0])); w.writeheader(); w.writerows(effects)
# factor means
summary={'cells':27,'maps':3,'factors':[.5,1,2],'global_search_seconds':sum(r['search_seconds'] for r in rows),
         'completed':sum(r['completed'] for r in rows),'failures':sum(r['failure'] is not None for r in rows),
         'actual_capacity':sorted({(r['seed'],r['factor'],r['capacity']) for r in rows}),
         'effects':effects}
summary['by_factor']={}
for fac in (.5,1,2):
 summary['by_factor'][str(float(fac))]={}
 for other in ('divide_then_plan','plan_then_divide'):
  xs=[e for e in effects if e['factor']==fac and e['comparison']==other+'_minus_cc']
  summary['by_factor'][str(float(fac))][other]={k:statistics.mean(e[k] for e in xs) for k in ('cost_diff','total_minus_work_diff','resupply_diff','departure_diff','payload_residual_diff','return_nonwork_diff','return_time_diff','nonreturn_nonwork_diff','maneuver_diff','scan_segments_diff')}
# descriptive interaction: change from factor1 to factor2/.5
for other in ('divide_then_plan','plan_then_divide'):
 summary[f'{other}_interaction_vs_1x']={f'{fac:g}x':summary['by_factor'][str(float(fac))][other]['cost_diff']-summary['by_factor']['1.0'][other]['cost_diff'] for fac in (.5,2.0)}
# per-map cost diff table
summary['per_map_cost_diff']=[{'seed':s,'factor':f,'other':o,'diff':by[(s,f,o)]['cost']-by[(s,f,'cc')]['cost']} for s in sorted({r['seed'] for r in rows}) for f in (.5,1.0,2.0) for o in ('divide_then_plan','plan_then_divide')]
(OUT/'payload_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
# plots
fig,axes=plt.subplots(1,3,figsize=(14,4.4),sharey=True)
for ax,seed in zip(axes,sorted({r['seed'] for r in rows})):
 for m,col in (('cc','#555'),('divide_then_plan','#1f4e9c'),('plan_then_divide','#2e8b57')):
  xs=[.5,1,2]; ys=[by[(seed,f,m)]['cost'] for f in xs]; ax.plot(xs,ys,'o-',label=LAB[m],color=col)
 ax.set_xscale('log'); ax.set_xticks([.5,1,2]); ax.set_xticklabels(['0.5×','1×','2×']); ax.set_title(f'map {seed}'); ax.set_xlabel('payload capacity'); ax.grid(alpha=.25)
axes[0].set_ylabel('best complete total time (s)'); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT/'cost_response.png',dpi=190,bbox_inches='tight'); plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,4.5))
for other,col,label in (('divide_then_plan','#1f4e9c','D→P−CC'),('plan_then_divide','#2e8b57','P→D−CC')):
 for seed,ls in zip(sorted({r['seed'] for r in rows}),(':','--','-.')):
  ax.plot([.5,1,2],[e['cost_diff'] for e in effects if e['seed']==seed and e['factor'] in (.5,1,2) and e['comparison']==other+'_minus_cc'],ls,color=col,alpha=.5,lw=1)
 ax.plot([.5,1,2],[summary['by_factor'][str(float(f))][other]['cost_diff'] for f in (.5,1,2)],'o-',color=col,lw=2,label=label+' mean')
ax.axhline(0,color='k',lw=.8); ax.set_xticks([.5,1,2]); ax.set_xticklabels(['0.5×','1×','2×']); ax.set_xlabel('payload capacity'); ax.set_ylabel('other − CC total (s)'); ax.set_title('M1 response screen (positive=CC faster)'); ax.legend(fontsize=7); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(OUT/'difference_response.png',dpi=190,bbox_inches='tight'); plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7))
for ax,metric,label in zip(axes.flat,('resupply_before_payload_ratio','resupplies','return_nonwork','maneuver'),('pre-resupply payload ratio','resupplies / map','return nonwork (s)','maneuver (s)')):
 for m,col in (('cc','#555'),('divide_then_plan','#1f4e9c'),('plan_then_divide','#2e8b57')):
  vals=[]
  for f in (.5,1,2):
   xs=[r[metric] for r in rows if r['factor']==f and r['method']==m and r[metric] is not None]; vals.append(statistics.mean(xs) if xs else None)
  ax.plot([.5,1,2],vals,'o-',label=LAB[m],color=col)
 ax.set_xticks([.5,1,2]); ax.set_xticklabels(['0.5×','1×','2×']); ax.set_title(label); ax.grid(alpha=.25)
axes.flat[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT/'mechanism_response.png',dpi=190,bbox_inches='tight'); plt.close(fig)
print(json.dumps({'cells':len(rows),'search_seconds':summary['global_search_seconds'],'by_factor':summary['by_factor'],'interaction':{k:summary[k] for k in summary if 'interaction' in k}},indent=2))
