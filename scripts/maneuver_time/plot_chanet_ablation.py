"""Raw A/B fixed-panel greedy curves with NN reference."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOC=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/chanet_ablation')
NN={'validation':3412.130187168398,'diagnostic':3514.9395502398124}
A_BEST=1060; B_BEST=760; FINAL=1200
COL={'A':'#1f4e9c','B':'#c4541a'}
for panel in ('validation','diagnostic'):
    data={}
    for arm in ('A','B'):
        rows=list(csv.DictReader((DOC/f'{arm}_{panel}_curve.csv').open()))
        data[arm]={int(r['update']):float(r['total_time']) for r in rows}
    fig,ax=plt.subplots(figsize=(11,5.8))
    for arm in ('A','B'):
        xs=sorted(data[arm]); ys=[data[arm][x] for x in xs]
        ax.plot(xs,ys,lw=1.35,marker='o',ms=2.6,color=COL[arm],
                label=f'{arm} raw greedy mean')
    ax.axhline(NN[panel],color='#555555',ls='--',lw=1.2,label=f'NN = {NN[panel]:.1f} s')
    ax.scatter([A_BEST],[data['A'][A_BEST]],s=80,facecolors='none',edgecolors='#1f4e9c',linewidths=2,zorder=6,
               label=f'A validation best u{A_BEST}: {data["A"][A_BEST]:.1f} s')
    ax.scatter([B_BEST],[data['B'][B_BEST]],s=80,facecolors='none',edgecolors='#c4541a',linewidths=2,zorder=6,
               label=f'B validation best u{B_BEST}: {data["B"][B_BEST]:.1f} s')
    ax.scatter([FINAL],[data['A'][FINAL]],s=65,marker='s',facecolors='none',edgecolors='#1f4e9c',linewidths=1.6,zorder=6)
    ax.scatter([FINAL],[data['B'][FINAL]],s=65,marker='s',facecolors='none',edgecolors='#c4541a',linewidths=1.6,zorder=6)
    ax.annotate(f'A u{A_BEST}\n{data["A"][A_BEST]:.1f}',(A_BEST,data['A'][A_BEST]),xytext=(-8,12),textcoords='offset points',fontsize=8,color='#1f4e9c',ha='right')
    ax.annotate(f'B u{B_BEST}\n{data["B"][B_BEST]:.1f}',(B_BEST,data['B'][B_BEST]),xytext=(-8,-16),textcoords='offset points',fontsize=8,color='#c4541a',ha='right')
    ax.annotate(f'A u1200\n{data["A"][FINAL]:.1f}',(FINAL,data['A'][FINAL]),xytext=(-8,10),textcoords='offset points',fontsize=8,color='#1f4e9c',ha='right')
    ax.annotate(f'B u1200\n{data["B"][FINAL]:.1f}',(FINAL,data['B'][FINAL]),xytext=(-8,-18),textcoords='offset points',fontsize=8,color='#c4541a',ha='right')
    ax.set_xlim(-15,1215); ax.set_xlabel('training update')
    ax.set_ylabel('model-estimated total greedy time (s)')
    ax.set_title(f'CHANet A vs B — fixed {panel} panel (raw points)')
    ax.grid(alpha=.25); ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(DOC/f'{panel}_AB_curve.png',dpi=190,bbox_inches='tight'); plt.close(fig)
    print(DOC/f'{panel}_AB_curve.png')
