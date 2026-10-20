"""Verify saved scientific outputs, holdout independence and prediction behavior."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'tmp/review_deps').exists():sys.path.insert(0,str(ROOT/'tmp/review_deps'))
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
sys.path.insert(0,str(ROOT/'scripts'))
from predict import predict

data=ROOT/'output/data';models=ROOT/'output/models'
d=pd.read_csv(data/'condenser_analysis.csv')
splits=pd.read_csv(data/'split_assignments.csv')
audit=json.loads((data/'data_audit.json').read_text())
meta=json.loads((models/'model_metadata.json').read_text())
test=splits.split.eq('test')
assert set(splits.loc[test,'condition_id']).isdisjoint(splits.loc[~test,'condition_id'])
assert d[['source_file','condition_id']].equals(splits[['source_file','condition_id']])
assert len(d)==audit['retained_rows'] and len(d)==350
assert len(pd.read_csv(data/'excluded_records.csv'))==1
assert hashlib.sha256((ROOT/'ASHRAE RP-1043.zip').read_bytes()).hexdigest()==audit['archive_sha256']
model=joblib.load(models/'evaluation_model_train_only.joblib')
y=d.loc[test,'recorded_approach_K'];pred=model['model'].predict(d.loc[test,model['features']])
for name,val in [('MAE_K',mean_absolute_error(y,pred)),('RMSE_K',np.sqrt(mean_squared_error(y,pred))),('R2',r2_score(y,pred))]:
 np.testing.assert_allclose(val,meta['rf_heldout'][name],rtol=1e-10,atol=1e-10)
saved=pd.read_csv(data/'heldout_predictions.csv')
np.testing.assert_allclose(pred,saved.predicted_approach_K)
assert not any(f in model['features'] for f in ['water_out_C','refrigerant_sat_C','COP','condition_id','source_file','recorded_approach_K'])
# Independent spot check from an actual source operating point in SI.
r=d.loc[(d.source_file=='normal1.xls')&(d.condition_id==1)].iloc[0]
np.testing.assert_allclose(r.water_in_C,(85.6-32)*5/9,rtol=1e-9)
np.testing.assert_allclose(r.heat_duty_kW,107.5*3.5168528421,rtol=1e-9)
assert ((d.effectiveness>0)&(d.effectiveness<1)).all()
assert predict(100,50,1000,20,0)['limited_local_support']
try:predict(-1,24,191,6.7,0)
except ValueError:pass
else:raise AssertionError('Invalid flow accepted')
print('PASS: source integrity, row accounting, grouped split, saved-model metrics, saved predictions, target leakage exclusions, independent unit spot checks, thermal bounds and inference validation.')
