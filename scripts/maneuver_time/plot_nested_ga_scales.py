from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga'); R=BASE/'scale_report_seed42'; D=json.loads((R/'summary.json').read_text())
# grouped control comparison per scale
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,(scale,label) in zip(axes,(('independent15','15 fields'),('independent20','20 fields'))):
 d=D[scale]; models=['GA','NN','GA init','A10 best','A10 final']
 vals=[d['ga_mean'],d['controls_mean']['original_NN']['total'],d['controls_mean']['ga_initializer']['total'],d['controls_mean']['A10_u1060']['total'],d['controls_mean']['A10_u1200']['total']]
 bars=ax.bar(models,vals,color=['#2e8b57','#777','#aaa','#1f4e9c','#4a7fc1'])
 for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+8,f'{v:.1f}',ha='center',fontsize=7)
 ax.set_title(f'{label} independent panel (n=30)'); ax.set_ylabel('total time (s)'); ax.grid(axis='y',alpha=.25); ax.tick_params(axis='x',rotation=20)
fig.tight_layout(); fig.savefig(R/'control_comparison.png',dpi=190,bbox_inches='tight'); plt.close(fig)
# progress grids
for scale in ('independent15','independent20'):
 rows=list(csv.DictReader((R/f'{scale}_progress.csv').open())); data={}
 for r in rows: data.setdefault(int(r['instance_seed']),[]).append(r)
 seeds=sorted(data); fig,axes=plt.subplots(5,6,figsize=(14,10.5),sharex=True)
 for ax,seed in zip(axes.flat,seeds):
  rr=sorted(data[seed],key=lambda z:int(z['evaluation']))
  ax.step([int(x['evaluation']) for x in rr],[float(x['incumbent']) for x in rr],where='post',color='#2e8b57')
  ax.set_title(str(seed),fontsize=7); ax.tick_params(labelsize=6); ax.grid(alpha=.2)
 fig.suptitle(f'{scale} seed42: incumbent vs candidate evaluations')
 fig.tight_layout(); fig.savefig(R/f'{scale}_progress.png',dpi=160,bbox_inches='tight'); plt.close(fig)
# quality vs search time
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,(scale,label) in zip(axes,(('independent15','15 fields'),('independent20','20 fields'))):
 d=D[scale]; xs=[t['search_seconds'] for t in d['trials']]; ys=[t['cost'] for t in d['trials']]
 ax.scatter(xs,ys,c='#2e8b57',s=28,label='GA instance')
 for key,val,color in (('NN',d['controls_mean']['original_NN']['total'],'#777'),('GA init',d['controls_mean']['ga_initializer']['total'],'#aaa'),('A10 best',d['controls_mean']['A10_u1060']['total'],'#1f4e9c'),('A10 final',d['controls_mean']['A10_u1200']['total'],'#4a7fc1')):
  ax.axhline(val,color=color,ls='--',lw=1,label=key)
 ax.set_title(f'{label}: quality vs search wall'); ax.set_xlabel('search wall (s)'); ax.set_ylabel('best total time (s)'); ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=2)
fig.tight_layout(); fig.savefig(R/'quality_vs_search_time.png',dpi=190,bbox_inches='tight'); plt.close(fig)
# 10/15/20 comparison table as grouped bars
ten=json.loads(Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga/independent10/report_seed42/summary.json').read_text())
rows=[('10',ten['ga_mean'],ten['controls_mean']['A10_u1060']['total'],ten['controls_mean']['original_NN']['total']),
      ('15',D['independent15']['ga_mean'],D['independent15']['controls_mean']['A10_u1060']['total'],D['independent15']['controls_mean']['original_NN']['total']),
      ('20',D['independent20']['ga_mean'],D['independent20']['controls_mean']['A10_u1060']['total'],D['independent20']['controls_mean']['original_NN']['total'])]
fig,ax=plt.subplots(figsize=(8,4.7)); import numpy as np
x=np.arange(len(rows)); w=.25
for i,(lab,color) in enumerate((('GA', '#2e8b57'),('A10 u1060','#1f4e9c'),('NN','#777'))):
 vals=[r[i+1] for r in rows]; ax.bar(x+(i-1)*w,vals,w,label=lab,color=color)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_xlabel('fields'); ax.set_ylabel('mean total time (s)'); ax.set_title('Archived 10-field and new 15/20-field panels'); ax.legend(); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); fig.savefig(R/'scale_comparison.png',dpi=190,bbox_inches='tight'); plt.close(fig)
print('done')
