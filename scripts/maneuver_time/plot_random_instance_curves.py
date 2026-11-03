"""Plot unsmoothed fixed-panel greedy curves for the random-instance run."""
from pathlib import Path
import csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOC=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/episode_rollout_random_instances')
BEST=1060
FINAL=1200
NN={'diagnostic':3514.9395502398124,'validation':3412.130187168398}
COL={'diagnostic':'#1b6ca8','validation':'#2e8b57'}
TITLE={'diagnostic':'Fixed diagnostic panel (train10, 10 instances)',
       'validation':'Fixed validation panel (valid30, 30 instances)'}
for name in ('diagnostic','validation'):
    rows=list(csv.DictReader((DOC/f'{name}_curve.csv').open()))
    rows.sort(key=lambda r:int(r['update']))
    x=[int(r['update']) for r in rows]; y=[float(r['total_time']) for r in rows]
    fig,ax=plt.subplots(figsize=(10.5,5.6))
    ax.plot(x,y,color=COL[name],lw=1.35,marker='o',ms=2.8,label='greedy mean (raw points)')
    ax.axhline(NN[name],color='#555555',ls='--',lw=1.2,label=f'NN = {NN[name]:.1f} s')
    selected={int(r['update']):float(r['total_time']) for r in rows}
    ax.scatter([BEST],[selected[BEST]],s=85,facecolors='none',edgecolors='#d95f02',linewidths=2,zorder=5,
               label=f'validation best u{BEST}: {selected[BEST]:.1f} s')
    ax.scatter([FINAL],[selected[FINAL]],s=85,marker='s',facecolors='none',edgecolors='#7b3294',linewidths=2,zorder=5,
               label=f'final u{FINAL}: {selected[FINAL]:.1f} s')
    ax.annotate(f'u{BEST}\n{selected[BEST]:.1f} s',(BEST,selected[BEST]),textcoords='offset points',xytext=(-8,12),
                fontsize=8,color='#d95f02',ha='right')
    ax.annotate(f'u{FINAL}\n{selected[FINAL]:.1f} s',(FINAL,selected[FINAL]),textcoords='offset points',xytext=(-8,-18),
                fontsize=8,color='#7b3294',ha='right')
    ax.set_xlim(-15,1215)
    ax.set_xlabel('training update')
    ax.set_ylabel('model-estimated total greedy time (s)')
    ax.set_title(TITLE[name])
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path=DOC/f'{name}_greedy_curve.png'
    fig.savefig(path,dpi=190,bbox_inches='tight')
    plt.close(fig)
    print(path)
