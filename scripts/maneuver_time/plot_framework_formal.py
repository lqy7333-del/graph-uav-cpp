import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/framework'); R=BASE/'formal_report'
S=json.loads((R/'summary.json').read_text()); labels={'cc':'CC','divide_then_plan':'D→P','plan_then_divide':'P→D'}
# method means
fig,ax=plt.subplots(figsize=(8,4.7)); x=np.arange(2); w=.25
for i,m in enumerate(('cc','divide_then_plan','plan_then_divide')):
 vals=[S[f]['aux'][i]['mean_total_time'] for f in ('10','20')]; ax.bar(x+(i-1)*w,vals,w,label=labels[m])
for i,m in enumerate(('cc','divide_then_plan','plan_then_divide')):
 for j,f in enumerate(('10','20')):
  v=S[f]['aux'][i]['mean_total_time']; ax.text(j+(i-1)*w,v+10,f'{v:.0f}',ha='center',fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(['10 fields','20 fields']); ax.set_ylabel('mean total time (s)'); ax.set_title('Framework formal comparison'); ax.legend(); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(R/'method_means.png',dpi=190,bbox_inches='tight'); plt.close(fig)
# paired difference distributions
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,f in zip(axes,('10','20')):
 vals=[]; names=[]
 for p in S[f]['primary']:
  vals.append(p['mean']); names.append(p['name'].replace('divide_then_plan','D→P').replace('plan_then_divide','P→D').replace(' - cc',' − CC'))
 ax.axhline(0,color='k',lw=1); ax.bar(names,vals,color=['#4472c4','#70ad47']); ax.set_ylabel('paired mean difference (s)'); ax.set_title(f'{f} fields (positive = CC faster)'); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(R/'paired_differences.png',dpi=190,bbox_inches='tight'); plt.close(fig)
# progress: best-so-far / final best vs evaluation, median per method/scale
extras={}
for run in ('formal10_run','formal20_run'):
 for p in (BASE/run).glob('*.json'):
  try: d=json.loads(p.read_text())
  except: continue
  if 'identity' in d and 'result' in d: extras[(d['identity']['id'],d['identity']['method'])]=d['result']
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
for ax,f in zip(axes,('10','20')):
 for m,col in (('cc','#555'),('divide_then_plan','#1f4e9c'),('plan_then_divide','#2e8b57')):
  curves=[]
  for (i,mm),res in extras.items():
   if not i.startswith('instance_1500' if f=='10' else 'instance_1800') or mm!=m: continue
   hist=res.get('history',[])
   final=res.get('best',{}).get('cost')
   if not final: continue
   pts=[]; best=np.inf
   for h in hist:
    if h.get('completed') and h.get('cost') is not None: best=min(best,h['cost'])
    if np.isfinite(best): pts.append((h['evaluation'],best/final))
   if pts: curves.append(pts)
  if not curves: continue
  grid=np.arange(1,386); ys=[]
  for pts in curves:
   # step best-so-far at integer evaluations
   vals=[]
   for e in grid:
    a=[v for k,v in pts if k<=e]; vals.append(a[-1] if a else np.nan)
   ys.append(vals)
  arr=np.asarray(ys,float); med=np.nanmedian(arr,axis=0); lo=np.nanpercentile(arr,25,axis=0); hi=np.nanpercentile(arr,75,axis=0)
  ax.plot(grid,med,label=labels[m],color=col); ax.fill_between(grid,lo,hi,color=col,alpha=.12)
 ax.set_xlabel('candidate evaluations'); ax.set_ylabel('best / final cost'); ax.set_title(f'{f} fields: incumbent progress (median, IQR)'); ax.legend(fontsize=7); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(R/'progress.png',dpi=180,bbox_inches='tight'); plt.close(fig)
# quality vs search wall
fig,ax=plt.subplots(figsize=(8,4.8))
for m,col in (('cc','#555'),('divide_then_plan','#1f4e9c'),('plan_then_divide','#2e8b57')):
 xs=[]; ys=[]
 for (i,mm),res in extras.items():
  if mm==m and res.get('best'): xs.append(res['search_seconds']); ys.append(res['best']['cost'])
 ax.scatter(xs,ys,s=14,alpha=.6,label=labels[m],color=col)
ax.set_xlabel('search wall time (s)'); ax.set_ylabel('best complete total time (s)'); ax.set_title('Quality vs online search wall'); ax.legend(); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(R/'quality_vs_search_time.png',dpi=190,bbox_inches='tight'); plt.close(fig)
# components
fig,ax=plt.subplots(figsize=(10,5)); width=.25; x=np.arange(2)
for i,m in enumerate(('cc','divide_then_plan','plan_then_divide')):
 vals=[S[f]['aux'][i]['mean_deducted'] for f in ('10','20')]; ax.bar(x+(i-1)*width,vals,width,label=labels[m])
ax.set_xticks(x); ax.set_xticklabels(['10 fields','20 fields']); ax.set_ylabel('mean total−work (s)'); ax.set_title('Deducted work cost'); ax.legend(); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(R/'deducted_cost.png',dpi=190,bbox_inches='tight'); plt.close(fig)
print('plots done')
