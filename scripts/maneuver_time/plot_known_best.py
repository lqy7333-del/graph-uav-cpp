from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
DOC=Path('/home/ps/Desktop/Graph_planning/graph-uav-overfit-persist/docs/fixed_instance_overfit')
d=json.loads((DOC/'known_best_per_instance.json').read_text())
rows=d['instances']; idx=np.arange(10); w=.26
fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
ax=axes[0]
ax.bar(idx-w, [r['initial'] for r in rows],w,label='initial (B42/240)',color='#ff8c42')
ax.bar(idx, [r['unified_best'] for r in rows],w,label='unified best (u780)',color='#2b6cb0')
ax.bar(idx+w, [r['best_known'] for r in rows],w,label='per-instance best known',color='#2e8b57')
ax.set_ylabel('total time (s)'); ax.set_title('Per-instance recorded minima vs a single unified model')
ax.legend(ncol=3,fontsize=8); ax.grid(axis='y',alpha=.25); ax.set_ylim(5500,9000)
ax=axes[1]
ax.bar(idx-.18,[r['unified_minus_best_known'] for r in rows],.34,label='unified − best known',color='#2e8b57')
ax.bar(idx+.18,[r['initial_minus_best_known'] for r in rows],.34,label='initial − best known',color='#ff8c42')
ax.axhline(0,color='black',lw=.7); ax.set_ylabel('time saved (s)')
ax.set_xlabel('fixed instance index / seed'); ax.set_xticks(idx,[f'{r["index"]}\n{r["seed"]}' for r in rows],fontsize=7)
ax.legend(fontsize=8); ax.grid(axis='y',alpha=.25)
fig.suptitle('Post-hoc per-instance minima over recorded executions (not a deployable policy)',fontsize=13)
fig.tight_layout()
fig.savefig(DOC/'known_best_per_instance.png',dpi=190,bbox_inches='tight')
plt.close(fig)
print(DOC/'known_best_per_instance.png')
