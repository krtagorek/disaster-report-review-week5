"""Standalone loss figure using separate plotting environment, no model inference."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'work/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
run=ROOT/'data/training/lora-v1'
report=json.loads((run/'loss-review.json').read_text())
logs=[json.loads(l) for l in (run/'losses.jsonl').read_text().splitlines()]
train=[r for r in logs if r['kind']=='train']
val=report['monitor_points']
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
ax.plot([r['iteration'] for r in train],[r['train_loss'] for r in train],color='#3366aa',label='Training (20-update mean)',linewidth=1.6)
ax.plot([r['iteration'] for r in val]+[600],[r['val_loss'] for r in val]+[report['final600_monitor_loss']],color='#ba5d20',marker='o',markersize=4,label='Fixed development monitor (240 reports)',linewidth=1.8)
ax.set(xlabel='Completed optimizer updates',ylabel='Label + EOS cross-entropy (nats/token)',title='Disaster Report Review — one bounded LoRA run',xlim=(0,610),ylim=(0,None))
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.18)
ax.legend(frameon=False,fontsize=9)
fig.text(.5,-.035,'40 development reports per class. No prompt/padding loss. Final checkpoint fixed at 600 updates.',ha='center',fontsize=9,color='#555555')
fig.savefig(run/'loss-curve.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print(run/'loss-curve.png')
