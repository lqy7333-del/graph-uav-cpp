from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga/report10_corrected_s1400000')
rows=list(csv.DictReader((R/'progress.csv').open())); inst=sorted({int(r['instance_seed']) for r in rows})
for mode,xkey,xlabel,out in (('eval','evaluation','candidate evaluations','progress_by_evaluation.png'),('wall','seconds','search seconds','progress_by_wallclock.png')):
 fig,axes=plt.subplots(5,1,figsize=(10,11.5),sharex=False)
 for ax,s in zip(axes,inst):
  rr=[r for r in rows if int(r['instance_seed'])==s]
  ax.step([float(r[xkey]) for r in rr],[float(r['incumbent']) for r in rr],where='post',color='#2e8b57',lw=1.4,label='incumbent')
  ax.set_title(f'instance {s}',fontsize=9); ax.set_ylabel('total time (s)'); ax.grid(alpha=.25); ax.legend(fontsize=7)
 axes[-1].set_xlabel(xlabel); fig.suptitle(f'Corrected nested GA incumbent progress by {xlabel} (raw)')
 fig.tight_layout(); fig.savefig(R/out,dpi=170,bbox_inches='tight'); plt.close(fig); print(R/out)
# controls bar
d=json.loads((R/'summary.json').read_text())
labels=['GA','NN','GA init','A10 best','A10 final']; vals=[d['ga_mean'],d['controls_mean']['original_NN']['total'],d['controls_mean']['ga_initializer']['total'],d['controls_mean']['A10_u1060']['total'],d['controls_mean']['A10_u1200']['total']]
colors=['#2e8b57','#777777','#aaa','#1f4e9c','#4a7fc1']
fig,ax=plt.subplots(figsize=(8,4.6)); bars=ax.bar(labels,vals,color=colors)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+8,f'{v:.1f}',ha='center',fontsize=8)
ax.set_ylabel('total time (s)'); ax.set_title('Corrected GA pilot: one seed across 5 instances'); ax.grid(axis='y',alpha=.25); ax.set_ylim(0,max(vals)*1.12)
fig.tight_layout(); fig.savefig(R/'control_comparison.png',dpi=190,bbox_inches='tight')
