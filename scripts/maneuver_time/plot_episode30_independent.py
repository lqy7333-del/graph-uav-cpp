from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/episode30_results/independent_eval/summary.json')
d=json.loads(p.read_text()); labels=['new30_best','new30_final','A10_best','A10_final','NN']
vals=[d['models'][x]['mean'] for x in labels]
colors=['#2e8b57','#5fae7c','#1f4e9c','#4a7fc1','#777777']
fig,ax=plt.subplots(figsize=(9,4.9)); bars=ax.bar(labels,vals,color=colors,width=.62)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+8,f'{v:.1f}',ha='center',fontsize=8)
ax.set_ylabel('model-estimated total greedy time (s)'); ax.set_title('Independent 30-field panel: frozen new30/A10/NN (n=30)')
ax.grid(axis='y',alpha=.25); ax.set_ylim(0,max(vals)*1.12); ax.text(.01,.97,'all 30/30 completed',transform=ax.transAxes,fontsize=8,va='top')
fig.tight_layout(); fig.savefig('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/episode30_results/independent_eval/independent_means.png',dpi=190,bbox_inches='tight')
