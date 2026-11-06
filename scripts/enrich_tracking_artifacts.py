import json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('.env')
import mlflow
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('outputs/full_assignment/evidence'); c=mlflow.MlflowClient()
runs=json.loads(Path('outputs/full_assignment/dagshub_instrument_runs.json').read_text())
fig,ax=plt.subplots(figsize=(8,4))
for i,r in enumerate(runs):
 rid=r['run_id']; d=out/'tracking'/rid; d.mkdir(parents=True,exist_ok=True)
 p=c.download_artifacts(rid,'history.json',str(d)); h=json.loads(Path(p).read_text())
 f,a=plt.subplots(figsize=(7,3.5)); a.plot(range(1,6),h['loss'],label='Training'); a.plot(range(1,6),h['val_loss'],label='Validation');a.set(xlabel='Epoch',ylabel='Binary cross entropy',title=f'23L-0727_A2_Ops | LR {r["learning_rate"]}, dropout {r["dropout"]}');a.legend(); f.tight_layout();f.savefig(d/'curves.png',dpi=150);plt.close(f)
 c.log_artifact(rid,str(d/'curves.png'))
 for s in ['train','val','test']:c.log_artifact(rid,f'outputs/23L-0727_A2_Ops/data_evidence/{s}.csv','split-manifests')
 c.log_artifact(rid,str(out/'tracking-requirements-lock.txt'),'environment')
 ax.plot(range(1,6),h['val_loss'],label=f'LR={r["learning_rate"]}, dropout={r["dropout"]} ({r["logging"]})')
ax.set(xlabel='Epoch',ylabel='Validation loss',title='Validation-only tracking demonstrations; champion unchanged');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/'tracking-comparison.png',dpi=170);plt.close(fig)
print('Six runs enriched with actual curves, split lists, and environment lock')
