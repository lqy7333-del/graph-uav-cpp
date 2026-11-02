"""Plot the 60-update actor-LR screen."""
from pathlib import Path
import csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
EXP=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/overfit20/short')
DOC=Path('/home/ps/Desktop/Graph_planning/graph-uav-overfit-persist/docs/fixed_instance_overfit')
DOC.mkdir(parents=True,exist_ok=True)
COL={'control':'#1b9e77','low':'#d95f02','start':'#555555'}
data={}
for name,path in (('control',EXP/'control_s42/learning_curve.csv'),('low',EXP/'lr3e5_s42/learning_curve.csv')):
 data[name]=list(csv.DictReader(path.open()))
summary=json.loads((EXP/'sampled_endpoints.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(14,5.2),gridspec_kw={'width_ratios':[1.55,1]})
ax=axes[0]
for name,label in (('control','control actor LR 3e-4'),('low','low actor LR 3e-5')):
 x=[int(r['update']) for r in data[name]]; y=[float(r['total_time']) for r in data[name]]
 ax.plot(x,y,marker='o',ms=4,lw=1.5,color=COL[name],label=label)
ax.axhline(7089.264913111205,color=COL['start'],ls='--',lw=1.2,label='frozen start 7089.3s')
ax.set_xlabel('fine-tuning update'); ax.set_ylabel('greedy mean total time (s)')
ax.set_title('Greedy evaluation every 10 updates')
ax.grid(alpha=.25); ax.set_xlim(-2,62); ax.legend(fontsize=7.5)
ax=axes[1]
labels=['start','control60','low60']; names=['frozen start','control 60','low 60']
means=[]; los=[]; his=[]
for k in labels:
 p=summary['points'][k]; means.append(p['sampled_mean']); lo,hi=p['conditional_ci95']; los.append(lo); his.append(hi)
colors=[COL['start'],COL['control'],COL['low']]
x=np.arange(3)
ax.bar(x,means,color=colors,width=.58)
ax.errorbar(x,means,yerr=[np.array(means)-np.array(los),np.array(his)-np.array(means)],fmt='none',ecolor='black',capsize=5)
ax.set_xticks(x,[f'{n}\n{m:.1f}s' for n,m in zip(names,means)],fontsize=7.5)
ax.set_ylabel('sampled mean total time (s)')
ax.set_title('8 sampled episodes per fixed instance')
ax.grid(axis='y',alpha=.25)
fig.suptitle('60-update actor-LR screen on the fixed 10x20 bank (in-sample)',fontsize=13)
fig.tight_layout()
fig.savefig(DOC/'short_lr_screen.png',dpi=190,bbox_inches='tight')
plt.close(fig)
print(DOC/'short_lr_screen.png')
