"""M0 figures and selected trajectory cases."""
import csv,json,pickle
from copy import deepcopy
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_env.baseline.nested_ga import action_key
from coverage_env.planner.coverage_plan import CoveragePlan
from coverage_env.planner.framework_execution import execute_region_plan, _execute_after_refill
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework'); M=BASE/'mechanism_m0'; F=BASE/'formal_report'
S=json.loads((M/'mechanism_summary.json').read_text()); W=list(csv.DictReader(open(M/'mechanism_instance.csv'))); FE=list(csv.DictReader(open(M/'features_enriched.csv'))); FFE=list(csv.DictReader(open(M/'field_features.csv')))
METHODS=('cc','divide_then_plan','plan_then_divide'); LAB={'cc':'CC','divide_then_plan':'D→P','plan_then_divide':'P→D'}
# cost component differences
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
for ax,scale in zip(axes,('10','20')):
 comps=['work','nonwork','maneuver','service']; others=['divide_then_plan','plan_then_divide']; x=np.arange(len(others)); bottomp=np.zeros(len(others)); bottomn=np.zeros(len(others))
 for c,color in zip(comps,['#4472c4','#ed7d31','#a5a5a5','#70ad47']):
  vals=[S[scale][f'{o}_minus_cc'][c]['mean'] for o in others]
  pos=np.maximum(vals,0); neg=np.minimum(vals,0)
  ax.bar(x,pos,bottom=bottomp,color=color,label=c); ax.bar(x,neg,bottom=bottomn,color=color); bottomp+=pos; bottomn+=neg
 ax.axhline(0,color='k',lw=1); ax.set_xticks(x); ax.set_xticklabels([LAB[o] for o in others]); ax.set_ylabel('mean component difference (s)'); ax.set_title(f'{scale} fields: D/P minus CC'); ax.grid(axis='y',alpha=.2); ax.legend(fontsize=7,ncol=2)
fig.tight_layout(); fig.savefig(M/'cost_decomposition.png',dpi=190,bbox_inches='tight'); plt.close(fig)
# paired structure
fig,axes=plt.subplots(2,3,figsize=(13,8)); metrics=['visits','trips','coverage_per_trip']
for row,scale in enumerate(('10','20')):
 for col,metric in enumerate(metrics):
  ax=axes[row,col]
  for other,color,marker in (('divide_then_plan','#1f4e9c','o'),('plan_then_divide','#2e8b57','s')):
   xx=[]; yy=[]
   for r in W:
    if r['scale']!=scale: continue
    a=r.get(f'cc_{metric}'); b=r.get(f'{other}_{metric}')
    if a not in (None,'','None') and b not in (None,'','None'): xx.append(float(a)); yy.append(float(b))
   ax.scatter(xx,yy,s=16,alpha=.7,color=color,marker=marker,label=LAB[other])
  ax.set_xlabel(f'CC {metric}'); ax.set_ylabel(f'other {metric}'); ax.set_title(f'{scale} {metric}',fontsize=8); ax.grid(alpha=.2)
  if row==0 and col==0: ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(M/'paired_structure.png',dpi=180,bbox_inches='tight'); plt.close(fig)
# features
fmap={r['instance_id']:r for r in FE}
for scale in ('10','20'):
 fig,axes=plt.subplots(2,3,figsize=(13,7)); feats=['total_area','field_area_cv','max_area','aspect_ratio_mean','depot_normalized_mean','area_over_qmax_max']
 for ax,name in zip(axes.flat,feats):
  for other,color,marker in (('divide_then_plan','#1f4e9c','o'),('plan_then_divide','#2e8b57','s')):
   xs=[]; ys=[]
   for r in W:
    if r['scale']!=scale: continue
    yv=r.get(f'{other}_minus_cc_total'); fv=fmap.get(r['instance_id'],{}).get(name)
    if yv not in (None,'','None') and fv not in (None,'','None'): xs.append(float(fv)); ys.append(float(yv))
   ax.scatter(xs,ys,s=14,alpha=.7,color=color,marker=marker,label=LAB[other])
  ax.axhline(0,color='k',lw=.8); ax.set_xlabel(name); ax.set_ylabel('D/P − CC total (s)'); ax.set_title(f'{scale} fields: {name}',fontsize=8); ax.grid(alpha=.2)
 axes.flat[0].legend(fontsize=6)
 fig.tight_layout(); fig.savefig(M/f'features_{scale}.png',dpi=170,bbox_inches='tight'); plt.close(fig)
# search budget
fig,axes=plt.subplots(2,3,figsize=(13,7))
for row,scale in enumerate(('10','20')):
 for col,metric in enumerate(('evaluation','search_seconds','full_rounds')):
  ax=axes[row,col]
  for other,color,marker in (('divide_then_plan','#1f4e9c','o'),('plan_then_divide','#2e8b57','s')):
   xs=[]; ys=[]
   for r in W:
    if r['scale']!=scale: continue
    yv=r.get(f'{other}_minus_cc_total'); xv=r.get(f'{other}_minus_cc_{metric}')
    if yv not in (None,'','None') and xv not in (None,'','None'): xs.append(float(xv)); ys.append(float(yv))
   ax.scatter(xs,ys,s=14,alpha=.7,color=color,marker=marker,label=LAB[other])
  ax.axhline(0,color='k',lw=.8); ax.set_xlabel(metric+' difference (other−CC)'); ax.set_ylabel('total difference (s)'); ax.set_title(f'{scale} fields: {metric}',fontsize=8); ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(M/'search_budget.png',dpi=170,bbox_inches='tight'); plt.close(fig)
# case routes
panel={}
for scale in (10,20):
 m=json.loads((BASE/f'formal{scale}_manifest.json').read_text()); panel[scale]=m
 cases=json.loads((M/'mechanism_summary.json').read_text())['cases']
 cases=[c for c in cases]
 outdir=M/'case_routes'; outdir.mkdir(exist_ok=True)
 with (M/'case_selection.csv').open('w',newline='') as fh:
  w=csv.DictWriter(fh,fieldnames=['scale','comparison','quantile','difference','instance_id','method_a','method_b']); w.writeheader()
  for c in cases: w.writerow({**c,'method_a':'cc','method_b':c['comparison'].replace('_minus_cc','')})
 def path_for(env,result,method):
  env=deepcopy(env); path=[]
  for row in result['best']['actions']:
   if method=='cc':
    acts=env.get_action_candidates(); aa=[a for a in acts if action_key(a)==tuple(row)]
    if len(aa)!=1: raise ValueError('case action mismatch')
    _,_,done,info=env.step(aa[0])
   else:
    plan=CoveragePlan.from_dict(row['plan'])
    _,_,done,info=(_execute_after_refill(env,plan,row['start_state_fingerprint']) if row.get('resupply_before') else execute_region_plan(env,plan))
   path.extend(info.get('path',[]))
  return env,path
 for idx,c in enumerate(cases):
  scale=int(c['scale']); m=json.loads((BASE/f'formal{scale}_manifest.json').read_text()); ent=next((x for x in m['instances'] if x['id']==c['instance_id']),None); assert ent
  env0=pickle.loads((BASE/ent['snapshot']).read_bytes());
  fig,ax=plt.subplots(figsize=(6.4,5.2))
  for f in env0.fields:
   xy=np.asarray(f.xy_polygon.exterior.coords); ax.fill(xy[:,0],xy[:,1],color='#eeeeee',edgecolor='#999',lw=.4,alpha=.8)
  depot=env0.depot.position; ax.scatter([depot[0]],[depot[1]],marker='*',s=80,color='k',zorder=5)
  colors={'cc':'#555','divide_then_plan':'#1f4e9c','plan_then_divide':'#2e8b57'}
  for method in METHODS:
   idxf=next(i for i,e in enumerate(m['instances']) if e['id']==c['instance_id']); rf=BASE/f'formal{scale}_run'/f'{idxf:04d}_{method}.json'
   res=json.loads(rf.read_text())['result']; _,path=path_for(env0,res,method); arr=np.asarray(path)
   ax.plot(arr[:,0],arr[:,1],color=colors[method],lw=1,alpha=.8,label=LAB[method])
  ax.set_aspect('equal'); ax.set_title(f"{c['scale']}f {LAB[c['comparison'].replace('_minus_cc','')]} q{int(c['quantile']*100)} diff {c['difference']:.1f}s",fontsize=8); ax.legend(fontsize=6); ax.grid(alpha=.2)
  fig.tight_layout(); fig.savefig(outdir/f"{scale}_{c['comparison']}_q{int(c['quantile']*100)}.png",dpi=150,bbox_inches='tight'); plt.close(fig)
print('plots and cases done')
