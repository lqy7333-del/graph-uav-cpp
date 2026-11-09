from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga/report10_s1400000')
d=json.loads((R/'summary.json').read_text())
labels=['GA mean','GA best-of-3*','NN','GA init','A10 best','A10 final']
vals=[sum(t['best_cost'] for t in d['trials'])/len(d['trials']),
      sum(r['ga_best_of_3'] for r in d['per_instance'])/5,
      d['controls_mean']['original_NN']['total_time'],
      d['controls_mean']['ga_initializer']['total_time'],
      d['controls_mean']['A10_u1060']['total_time'],
      d['controls_mean']['A10_u1200']['total_time']]
colors=['#2e8b57','#7fbf9a','#777777','#a7a7a7','#1f4e9c','#4a7fc1']
fig,ax=plt.subplots(figsize=(9,4.8)); bars=ax.bar(labels,vals,color=colors,width=.65)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+8,f'{v:.1f}',ha='center',fontsize=8)
ax.set_ylabel('model-estimated total greedy time (s)'); ax.set_title('Nested GA pilot (5 instances): controls and GA')
ax.grid(axis='y',alpha=.25); ax.set_ylim(0,max(vals)*1.12); plt.xticks(rotation=20,ha='right')
ax.text(.01,.97,'* best-of-3 uses 3x search budget',transform=ax.transAxes,fontsize=8,va='top')
fig.tight_layout(); fig.savefig(R/'control_comparison.png',dpi=190,bbox_inches='tight')
