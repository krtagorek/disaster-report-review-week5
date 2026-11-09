"""Submission-ready static comparisons from saved evaluation evidence only."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'work/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
run=ROOT/'data/training/lora-v1'
comparison=json.loads((run/'comparison.json').read_text())
r=comparison['reports']
labels=list(r['merged_raw']['per_class'])
short=['Flash flood','Tornado','Storm wind','Hail','Winter storm','Wildfire']
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
x=np.arange(len(labels)+2)
width=.25
for shift,key,name,color in [(-1,'keyword','Keyword','#737373'),(0,'base_raw','Base raw','#5894c6'),(1,'merged_raw','Merged LoRA raw','#bd6029')]:
    vals=[r[key]['per_class'][c]['f1'] for c in labels]+[r[key]['accuracy'],r[key]['macro_f1']]
    ax.bar(x+shift*width,vals,width,label=name,color=color)
ax.set_xticks(x,short+['Accuracy','Macro-F1'],rotation=25,ha='right')
ax.set(ylim=(0,1.12),ylabel='F1 / accuracy',title='Development comparison — 2,449 identical reports')
ax.axvline(5.5,color='#999999',linewidth=.8)
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False,ncol=3,loc='upper center',fontsize=9)
ax.grid(axis='y',alpha=.15)
fig.savefig(run/'development-comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained',sharey=True)
columns=labels+['INVALID']
for ax,key,name in zip(axes,['base_raw','adapter_raw','merged_raw'],['Base raw','Attached adapter raw','Reloaded merged raw']):
    counts=np.array([[r[key]['confusion_matrix'][t][p] for p in columns] for t in labels])
    fractions=counts/counts.sum(axis=1,keepdims=True)
    ax.imshow(fractions,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    for i in range(6):
        for j in range(7):
            ax.text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=9,color='white' if fractions[i,j]>.55 else '#222222')
    ax.set_xticks(range(7),['FF','Tor','Wind','Hail','Winter','Fire','Invalid'],rotation=40,ha='right')
    ax.set_yticks(range(6),short)
    ax.set(xlabel='Predicted category',title=f"{name}\nAccuracy {r[key]['accuracy']:.2%}, macro-F1 {r[key]['macro_f1']:.2%}")
axes[0].set_ylabel('Reference category')
fig.suptitle('Matched development confusion matrices — counts shown, color is row fraction',fontsize=13)
fig.savefig(run/'confusion-matrices.png',dpi=180,bbox_inches='tight');plt.close(fig)
print('Saved development-comparison.png and confusion-matrices.png')
