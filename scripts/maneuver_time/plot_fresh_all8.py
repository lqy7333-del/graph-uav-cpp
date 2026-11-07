from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/chanet_component_ablation/fresh_independent_eval/summary.json')
d=json.loads(p.read_text()); labels=['A_best','A_final','B_best','B_final','C_best','C_final','D_best','D_final','NN']
vals=[d['models'][x]['mean'] for x in labels]; colors=['#1f4e9c','#4a7fc1','#c4541a','#e07b39','#2e8b57','#5fae7c','#7b3294','#a56bc0','#777777']
fig,ax=plt.subplots(figsize=(11,5.2)); bars=ax.bar(labels,vals,color=colors,width=.68)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+7,f'{v:.1f}',ha='center',fontsize=7.5)
ax.set_ylabel('model-estimated total greedy time (s)'); ax.set_title('Fresh 10-field panel: all frozen A/B/C/D checkpoints and NN (n=20)')
ax.grid(axis='y',alpha=.25); ax.set_ylim(0,max(vals)*1.12); plt.xticks(rotation=25,ha='right')
ax.text(.01,.97,'all 20/20 completed',transform=ax.transAxes,fontsize=8,va='top')
fig.tight_layout(); fig.savefig('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/chanet_component_ablation/fresh_independent_eval/fresh_all8_means.png',dpi=190,bbox_inches='tight')
