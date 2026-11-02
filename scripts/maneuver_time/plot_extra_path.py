from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
DOC=Path('/home/ps/Desktop/Graph_planning/graph-uav-overfit-persist/docs/fixed_instance_overfit')
d=json.loads((DOC/'extra_path_summary.json').read_text()); rows=d['instances']; idx=np.arange(10); w=.26
fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
ax=axes[0]
ax.bar(idx-w,[r['initial_extra'] for r in rows],w,label='initial extra',color='#ff8c42')
ax.bar(idx,[r['unified_extra'] for r in rows],w,label='unified best extra (u780)',color='#2b6cb0')
ax.bar(idx+w,[r['best_known_extra'] for r in rows],w,label='per-instance best-known extra',color='#2e8b57')
ax.set_ylabel('extra path time (s)'); ax.set_title('Extra path cost = total time − necessary work base (median per instance)')
ax.legend(ncol=3,fontsize=8); ax.grid(axis='y',alpha=.25)
ax=axes[1]
uni=[100*r['initial_minus_unified_extra']/r['initial_extra'] for r in rows]
best=[100*r['initial_minus_best_extra']/r['initial_extra'] for r in rows]
ax.bar(idx-.18,uni,.34,label='unified best vs initial',color='#2b6cb0')
ax.bar(idx+.18,best,.34,label='per-instance best-known vs initial',color='#2e8b57')
ax.axhline(0,color='black',lw=.7); ax.set_ylabel('reduction in extra path time (%)')
ax.set_xlabel('fixed instance index / seed'); ax.set_xticks(idx,[f'{r["index"]}\n{r["seed"]}' for r in rows],fontsize=7)
ax.legend(fontsize=8); ax.grid(axis='y',alpha=.25)
fig.suptitle('How much of the extra path cost is optimized (in-sample records)',fontsize=13)
fig.tight_layout()
fig.savefig(DOC/'extra_path_costs.png',dpi=190,bbox_inches='tight')
plt.close(fig)
print(DOC/'extra_path_costs.png')
