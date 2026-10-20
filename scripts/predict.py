"""Predict approach in kelvin using SI inputs. Domain checks do not prove causal validity."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
if (ROOT/'tmp/review_deps').exists(): sys.path.insert(0,str(ROOT/'tmp/review_deps'))
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np


def predict(flow, inlet, load, evap_out, blockage, model_path=None):
    if flow<=0 or load<=0: raise ValueError('Flow and load must be positive.')
    if not 0<=blockage<1: raise ValueError('Blocked-tube fraction must lie in [0,1).')
    path=Path(model_path) if model_path else ROOT/'output/models/condenser_approach_model.joblib'
    bundle=joblib.load(path)
    vals=[flow,inlet,load,evap_out,blockage]
    if not np.isfinite(vals).all():raise ValueError('All inputs must be finite.')
    x=pd.DataFrame([vals],columns=bundle['features'])
    outside=[c for c in x if x[c].iloc[0]<bundle['feature_min'][c] or x[c].iloc[0]>bundle['feature_max'][c]]
    train=bundle['training_features'];scale=(train.max()-train.min()).replace(0,1)
    near=NearestNeighbors(n_neighbors=2).fit(train/scale)
    limit=float(np.quantile(near.kneighbors(train/scale)[0][:,1],.95))
    distance=float(near.kneighbors(x/scale,n_neighbors=1)[0][0,0])
    return {'predicted_approach_K':float(bundle['model'].predict(x)[0]),
            'outside_training_ranges':outside,'limited_local_support':bool(outside or distance>limit),
            'scope':bundle['scope'],
            'interpretation':'Prediction of approach, not efficiency or failure probability. No validated lab-rig transfer.'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--flow',type=float,required=True,help='Cooling-water mass flow in kg/s')
    p.add_argument('--inlet',type=float,required=True,help='Entering condenser water in C')
    p.add_argument('--load',type=float,required=True,help='Evaporator cooling load in kW')
    p.add_argument('--evap-out',type=float,required=True,help='Leaving evaporator water in C')
    p.add_argument('--blockage',type=float,default=0,help='Known blocked fraction of condenser tubes, 0 to 1')
    a=p.parse_args();print(json.dumps(predict(a.flow,a.inlet,a.load,a.evap_out,a.blockage),indent=2))
