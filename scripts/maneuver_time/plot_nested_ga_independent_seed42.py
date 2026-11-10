from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga/independent10/report_seed42')
s=json.loads((R/'summary.json').read_text())
# Cost comparison
labels=['GA seed42','NN','GA init','A10 best','A10 final']
vals=[s['ga_mean'],s['controls_mean']['original_NN']['total'],s['controls_mean']['ga_initializer']['total'],s['controls_mean']['A10_u1060']['total'],s['controls_mean']['A10_u1200']['total']]
fig,ax=plt.subplots(figsize=(8.6,4.8)); bars=ax.bar(labels,vals,color=['#2e8b57','#777','#aaa','#1f4e9c','#4a7fc1'])
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+8,f'{v:.1f}',ha='center',fontsize=8)
ax.set_ylabel('total time (s)'); ax.set_title('Independent 10-field panel (n=30, GA seed42)'); ax.grid(axis='y',alpha=.25); ax.set_ylim(0,max(vals)*1.12)
fig.tight_layout(); fig.savefig(R/'control_comparison.png',dpi=190,bbox_inches='tight'); plt.close(fig)
# Components
models=['GA','NN','GA init','A10 best','A10 final']; comps=[('work','#8fbcd4'),('nonwork','#f2a65a'),('maneuver','#c44e52'),('service','#8172b2')]
rows={m:{'work':0,'nonwork':0,'maneuver':0,'service':0} for m in models}
rows['GA']={'work':sum(t['work'] for t in s['trials'])/30,'nonwork':sum(t['nonwork'] for t in s['trials'])/30,'maneuver':sum(t['maneuver'] for t in s['trials'])/30,'service':0}
for m,key in (('NN','original_NN'),('GA init','ga_initializer'),('A10 best','A10_u1060'),('A10 final','A10_u1200')):
 c=s['controls_mean'][key]; rows[m]={'work':c.get('work'),'nonwork':0,'maneuver':0,'service':0}
# control components not in summary; read rows
crows=list(csv.DictReader(open('/home/ps/Desktop/Graph_planning/graph-uav-experiments/nested_ga/independent10/controls/rows.csv')))
for m,key in (('NN','original_NN'),('GA init','ga_initializer'),('A10 best','A10_u1060'),('A10 final','A10_u1200')):
 rr=[r for r in crows if r['model']==key]
 rows[m]={k:sum(float(r['time_'+k]) for r in rr)/len(rr) for k in ('work_base','nonwork_base','maneuver_extra','service')}
# normalize keys
width=.16; x=range(len(models)); fig,ax=plt.subplots(figsize=(10,5)); bottom=[0]*len(models)
for i,(lab,col) in enumerate(comps):
 keys={'work':'work_base','nonwork':'nonwork_base','maneuver':'maneuver_extra','service':'service'}
 vals=[]
 for m in models:
  if m=='GA': vals.append({'work':sum(t['work'] for t in s['trials'])/30,'nonwork':sum(t['nonwork'] for t in s['trials'])/30,'maneuver':sum(t['maneuver'] for t in s['trials'])/30,'service':0}[lab])
  else: vals.append(rows[m][keys[lab]])
 ax.bar(models,vals,bottom=bottom,color=col,label=lab,width=.62); bottom=[a+b for a,b in zip(bottom,vals)]
ax.set_ylabel('time (s)'); ax.set_title('Cost components (mean per instance)'); ax.legend(ncol=4,fontsize=8); ax.grid(axis='y',alpha=.2)
fig.tight_layout(); fig.savefig(R/'cost_components.png',dpi=190,bbox_inches='tight'); plt.close(fig)
# Progress grid
pr=csv.DictReader(open(R/'progress.csv')); data={}
for r in pr: data.setdefault(int(r['instance_seed']),[]).append(r)
seeds=sorted(data); fig,axes=plt.subplots(5,6,figsize=(14,11),sharex=True)
for ax,seed in zip(axes.flat,seeds):
 rr=sorted(data[seed],key=lambda z:int(z['evaluation']))
 ax.step([int(x['evaluation']) for x in rr],[float(x['incumbent']) for x in rr],where='post',color='#2e8b57')
 ax.set_title(str(seed),fontsize=8); ax.tick_params(labelsize=6); ax.grid(alpha=.2)
 if ax in axes[:,0]: ax.set_ylabel('incumbent s',fontsize=7)
 if ax in axes[-1,:]: ax.set_xlabel('evaluations',fontsize=7)
fig.suptitle('Independent 10-field GA seed42: incumbent vs candidate evaluations')
fig.tight_layout(); fig.savefig(R/'progress_by_evaluation.png',dpi=160,bbox_inches='tight'); plt.close(fig)
