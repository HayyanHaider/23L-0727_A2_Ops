import json,os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('.env')
import mlflow, numpy as np
from tensorflow import keras
out=Path('outputs/full_assignment/evidence'); c=mlflow.MlflowClient()
records=[]
for eid in ['0','1']:
 for run in c.search_runs([eid],max_results=100):
  metrics={k:[dict(step=x.step,value=x.value,timestamp=x.timestamp) for x in c.get_metric_history(run.info.run_id,k)] for k in run.data.metrics if k in ['loss','val_loss','epoch_loss','epoch_val_loss','accuracy','val_accuracy']}
  records.append(dict(run_id=run.info.run_id,experiment_id=eid,status=run.info.status,name=run.data.tags.get('mlflow.runName'),params=run.data.params,tags=run.data.tags,metrics=run.data.metrics,history=metrics,artifacts=[x.path for x in c.list_artifacts(run.info.run_id)]))
versions=[dict(version=x.version,run_id=x.run_id,status=x.status,aliases=x.aliases) for x in c.search_model_versions("name='23L-0727_A2_Ops'")]
(out/'remote-mlflow-audit.json').write_text(json.dumps(dict(project='23L-0727_A2_Ops',runs=records,versions=versions,champion=c.get_model_version_by_alias('23L-0727_A2_Ops','champion').version),indent=2))
a=keras.models.load_model('models/champion.keras',compile=False); b=keras.models.load_model('outputs/full_assignment/dagshub_registry_export/champion.keras',compile=False)
assert len(a.weights)==len(b.weights)
assert all(np.array_equal(x.numpy(),y.numpy()) for x,y in zip(a.weights,b.weights))
(out/'registry-parity.json').write_text(json.dumps(dict(weights_equal=True,arrays=len(a.weights),alias='champion',version='2',release_sha256='086258a9016fd9db5d7aa664900250921ef5596d2fb9ce7a94de21c4653d86a5'),indent=2))
print('Verified',len(records),'remote runs; registry weights match original champion')
