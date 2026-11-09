from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DOC=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/episode30_results')
REF={'validation':{'NN':11349.2,'A10 u1060':11059.3,'A10 u1200':11075.3},
     'diagnostic':{'NN':10929.3,'A10 u1060':10716.5,'A10 u1200':10657.3}}
BEST={'validation':(1060,11020.5),'diagnostic':(1060,10637.8)}
FINAL={'validation':(1200,11104.1),'diagnostic':(1200,10759.4)}
for panel in ('validation','diagnostic'):
 rows=list(csv.DictReader((DOC/f'new_{panel}_curve.csv').open())); rows.sort(key=lambda r:int(r['update']))
 xs=[int(r['update']) for r in rows]; ys=[float(r['total_time']) for r in rows]
 fig,ax=plt.subplots(figsize=(10.8,5.6))
 ax.plot(xs,ys,color='#2e8b57',lw=1.4,marker='o',ms=2.7,label='30-field trained CHANet (raw)')
 for label,val,color,ls in (('NN',REF[panel]['NN'],'#888888',':'),('A10 u1060',REF[panel]['A10 u1060'],'#1f4e9c','--'),('A10 u1200',REF[panel]['A10 u1200'],'#7b3294','-.')):
  ax.axhline(val,color=color,ls=ls,lw=1.1,label=f'{label}: {val:.1f} s')
 b=BEST[panel]; f=FINAL[panel]
 ax.scatter([b[0]],[b[1]],s=85,facecolors='none',edgecolors='#d95f02',linewidths=2,zorder=6,label=f'validation best u{b[0]}: {b[1]:.1f} s')
 ax.scatter([f[0]],[f[1]],s=75,marker='s',facecolors='none',edgecolors='#555555',linewidths=1.8,zorder=6,label=f'final u{f[0]}: {f[1]:.1f} s')
 ax.set_xlim(-15,1215); ax.set_xlabel('training update'); ax.set_ylabel('model-estimated total greedy time (s)')
 ax.set_title(f'30-field CHANet vs frozen references — fixed {panel} panel (raw points)')
 ax.grid(alpha=.25); ax.legend(fontsize=7.5,ncol=2)
 fig.tight_layout(); fig.savefig(DOC/f'new_{panel}_vs_references.png',dpi=190,bbox_inches='tight'); plt.close(fig)
 print(DOC/f'new_{panel}_vs_references.png')
