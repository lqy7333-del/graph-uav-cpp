from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
DOC=Path('/home/ps/Desktop/Graph_planning/graph-uav-overfit-persist/docs/fixed_instance_overfit')
d=json.loads((DOC/'search_final_table.json').read_text()); rows=d['rows']; idx=np.arange(10); w=.26
fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True)
ax=axes[0]
ax.bar(idx-w,[r['initial'] for r in rows],w,label='initial',color='#ff8c42')
ax.bar(idx,[r['current'] for r in rows],w,label='current saved (control_s43)',color='#2b6cb0')
ax.bar(idx+w,[r['search'] for r in rows],w,label='search best',color='#2e8b57')
ax.set_ylabel('total time (s)'); ax.set_title('Replayable rollout search on fixed instances')
ax.legend(fontsize=8); ax.grid(axis='y',alpha=.25)
ax=axes[1]
ax.bar(idx,[r['improvement'] for r in rows],.55,color='#2e8b57')
for i,r in enumerate(rows):
 if r['improvement']>0: ax.text(i,r['improvement']+2,f'{r["improvement"]:.0f}',ha='center',fontsize=7)
ax.axhline(0,color='black',lw=.7); ax.set_ylabel('current − search (s)')
ax.set_xlabel('instance index / seed'); ax.set_xticks(idx,[f'{r["index"]}\n{r["seed"]}' for r in rows],fontsize=7)
ax.grid(axis='y',alpha=.25)
fig.suptitle('40-completion/instance search: 9/10 improved; mean 67.4 s (0.96%)',fontsize=12)
fig.tight_layout(); fig.savefig(DOC/'search_formal.png',dpi=190,bbox_inches='tight'); print(DOC/'search_formal.png')
