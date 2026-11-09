from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga/report10_s1400000')
rows=list(csv.DictReader((R/'progress.csv').open()))
inst=sorted({int(r['instance_seed']) for r in rows}); seeds=sorted({int(r['search_seed']) for r in rows})
colors={42:'#1f4e9c',43:'#c4541a',44:'#2e8b57'}
for mode,xkey,xlabel,out in (('eval','evaluation','candidate evaluations','progress_by_evaluation.png'),('wall','seconds','search seconds','progress_by_wallclock.png')):
 fig,axes=plt.subplots(5,1,figsize=(10,12),sharex=False)
 for ax,s in zip(axes,inst):
  for g in seeds:
   rr=[r for r in rows if int(r['instance_seed'])==s and int(r['search_seed'])==g]
   ax.plot([float(r[xkey]) for r in rr],[float(r['incumbent']) for r in rr],color=colors[g],lw=1.1,label=f'GA {g}')
  ax.set_title(f'instance {s}',fontsize=9); ax.set_ylabel('incumbent total (s)'); ax.grid(alpha=.25); ax.legend(fontsize=7)
 axes[-1].set_xlabel(xlabel); fig.suptitle(f'Nested GA incumbent progress by {xlabel} (raw)')
 fig.tight_layout(); fig.savefig(R/out,dpi=170,bbox_inches='tight'); plt.close(fig)
 print(R/out)
