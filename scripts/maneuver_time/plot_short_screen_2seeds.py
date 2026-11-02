from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
DOC=Path('/home/ps/Desktop/Graph_planning/graph-uav-overfit-persist/docs/fixed_instance_overfit')
COL={'control':'#1b9e77','low':'#d95f02','start':'#555555'}
fig,axes=plt.subplots(2,2,figsize=(13,9))
for col,seed in enumerate((42,43)):
 for name,label in (('control','control 3e-4'),('low','low 3e-5')):
  p=DOC/f'short_{"control" if name=="control" else "low"}_curve{"" if seed==42 else "_s43"}.csv'
  rr=list(csv.DictReader(p.open()))
  ax=axes[0,col]; ax.plot([int(r['update']) for r in rr],[float(r['total_time']) for r in rr],
      marker='o',ms=3.5,lw=1.4,color=COL[name],label=label)
 ax=axes[0,col]; ax.axhline(7089.264913111205,color=COL['start'],ls='--',lw=1.1,label='start')
 ax.set_title(f'seed {seed}: greedy'); ax.grid(alpha=.25); ax.set_xlim(-2,62)
 ax.set_xlabel('update'); ax.set_ylabel('total time (s)'); ax.legend(fontsize=7)
 ax=axes[1,col]
 summ=json.loads((DOC/f'short_sampled_endpoints{"" if seed==42 else "_s43"}.json').read_text())
 labels=['start','control60','low60']; names=['start','control','low']; colors=[COL['start'],COL['control'],COL['low']]
 means=[]; los=[]; his=[]
 for k in labels:
  v=summ['points'][k]; means.append(v['sampled_mean']); lo,hi=v['conditional_ci95']; los.append(lo); his.append(hi)
 x=np.arange(3); ax.bar(x,means,color=colors,width=.58)
 ax.errorbar(x,means,yerr=[np.array(means)-np.array(los),np.array(his)-np.array(means)],fmt='none',ecolor='black',capsize=4)
 ax.set_xticks(x,[f'{n}\n{m:.1f}' for n,m in zip(names,means)],fontsize=7)
 ax.set_title(f'seed {seed}: 8-sample mean'); ax.set_ylabel('sampled time (s)'); ax.grid(axis='y',alpha=.25)
fig.suptitle('Actor-LR screen replicated on seed 43 (fixed ten instances, in-sample)',fontsize=13)
fig.tight_layout()
fig.savefig(DOC/'short_lr_screen_2seeds.png',dpi=190,bbox_inches='tight')
plt.close(fig)
print(DOC/'short_lr_screen_2seeds.png')
