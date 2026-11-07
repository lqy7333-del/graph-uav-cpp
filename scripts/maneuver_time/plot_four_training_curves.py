"""Raw A/B/C/D fixed-panel curves in one two-panel comparison figure."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOC=Path('/home/ps/Desktop/Graph_planning/graph-uav-episode-10fields/docs/chanet_component_ablation')
COL={'A':'#1f4e9c','B':'#c4541a','C':'#2e8b57','D':'#7b3294'}
SEL={'A':1060,'B':760,'C':1020,'D':900}
NN={'validation':3412.130187168398,'diagnostic':3514.9395502398124}
fig,axes=plt.subplots(2,1,figsize=(11.5,9.4),sharex=True)
for ax,panel in zip(axes,('validation','diagnostic')):
    for arm in 'ABCD':
        rows=list(csv.DictReader((DOC/f'{arm}_{panel}_curve.csv').open()))
        rows.sort(key=lambda r:int(r['update']))
        xs=[int(r['update']) for r in rows]; ys=[float(r['total_time']) for r in rows]
        ax.plot(xs,ys,color=COL[arm],lw=1.25,marker='o',ms=2.1,
                label=f'{arm} raw greedy')
        bi=SEL[arm]; by={int(r['update']):float(r['total_time']) for r in rows}[bi]
        ax.scatter([bi],[by],s=62,facecolors='none',edgecolors=COL[arm],linewidths=1.8,zorder=5)
        ax.scatter([1200],[ys[-1]],s=42,marker='s',facecolors='none',edgecolors=COL[arm],linewidths=1.4,zorder=5)
    ax.axhline(NN[panel],color='#555',ls='--',lw=1.15,label=f'NN {NN[panel]:.1f}')
    ax.set_ylabel('model-estimated greedy total (s)')
    ax.set_title('Fixed validation panel (valid30, n=30)' if panel=='validation' else 'Fixed diagnostic panel (train10, n=10)')
    ax.grid(alpha=.25)
    from matplotlib.lines import Line2D
    handles,labels=ax.get_legend_handles_labels()
    handles += [Line2D([],[],marker='o',ls='',markerfacecolor='none',markeredgecolor='#444',markersize=7),
                Line2D([],[],marker='s',ls='',markerfacecolor='none',markeredgecolor='#444',markersize=6)]
    labels += ['validation-selected checkpoint','final u1200']
    ax.legend(handles,labels,fontsize=7.2,ncol=3,loc='upper right')
    ax.set_ylim(3220 if panel=='validation' else 3350,
                3870 if panel=='validation' else 3970)
axes[-1].set_xlabel('training update'); axes[-1].set_xlim(-15,1215)
fig.suptitle('CHANet training curves: A full graph vs B/C/D component ablations (raw points)',fontsize=12)
fig.tight_layout(rect=(0,0,1,.96))
out=DOC/'ABCD_combined_curves.png'; fig.savefig(out,dpi=190,bbox_inches='tight'); plt.close(fig)
print(out)
