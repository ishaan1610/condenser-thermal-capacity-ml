"""Reproduce the condenser study from the supplied RP-1043 ZIP. No synthetic fallback."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / 'tmp/review_deps').exists():
    sys.path.insert(0, str(ROOT / 'tmp/review_deps'))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xlrd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CP = 4.186  # kJ / (kg K), constant-water approximation
RHO = 997.0  # kg/m3, approximate water density
GPM_TO_M3_S = 0.003785411784 / 60
TON_TO_KW = 3.5168528421
FEATURES = ['flow_kg_s', 'water_in_C', 'evaporator_load_kW',
            'evaporator_out_C', 'blocked_tube_fraction']
TARGET = 'recorded_approach_K'
COLORS = {'normal': '#256c85', 'reduced_flow': '#d77c25', 'tube_blockage': '#89569f'}


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def selections():
    base = 'Chiller Data/'
    names = [(base+'Benchmark Tests/'+s+'.xls', 'normal', 0., 0.)
             for s in ['normal', 'normal1', 'normal2', 'normal cf']]
    names += [(base+f'Reduced condenser water flow/fwc{s}.xls', 'reduced_flow', s/100, 0.)
              for s in [10, 20, 30, 40]]
    names += [(base+f'Condenser fouling/cf{s}.xls', 'tube_blockage', 0., tubes/164)
              for s, tubes in [(6, 10), (12, 20), (20, 33), (30, 49), (45, 74)]]
    return names


def load_data(archive, output):
    frames, inventory = [], []
    with ZipFile(archive) as z:
        for name, family, reduction, blockage in selections():
            book = xlrd.open_workbook(file_contents=z.read(name))
            sheet = book.sheet_by_name('Steady States')
            headers = sheet.row_values(0)
            frame = pd.DataFrame([sheet.row_values(i) for i in range(1, sheet.nrows)], columns=headers)
            frame['source_file'] = Path(name).name
            frame['fault_family'] = family
            frame['nominal_flow_reduction'] = reduction
            frame['blocked_tube_fraction'] = blockage
            frame['source_excel_row'] = np.arange(2, len(frame)+2)
            frames.append(frame)
            inventory.append({'archive_path': name, 'sheet': sheet.name, 'records': len(frame),
                              'family': family, 'nominal_flow_reduction': reduction,
                              'blocked_tube_fraction': blockage})
        notes = xlrd.open_workbook(file_contents=z.read('Chiller Data/read me first.xls'))
        note_sheet = notes.sheet_by_index(0)
        pd.DataFrame([note_sheet.row_values(i) for i in range(1, note_sheet.nrows)],
                     columns=note_sheet.row_values(0)).to_csv(output/'source_test_notes.csv', index=False)
    pd.DataFrame(inventory).to_csv(output/'source_inventory.csv', index=False)
    return pd.concat(frames, ignore_index=True)


def prepare(raw, output):
    required = ['Test', 'TCI', 'TCO', 'FWC', 'Cond Tons', 'TRC', 'TCA', 'TEO',
                'Evap Tons', 'kW', 'Heat Balance%']
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    d = raw.copy()
    for col in required:
        d[col] = pd.to_numeric(d[col], errors='coerce')
    valid = np.isfinite(d[required]).all(axis=1)
    valid &= (d.FWC > 0) & (d['Cond Tons'] > 0) & (d['Evap Tons'] > 0) & (d.kW > 0)
    valid &= (d.TCO > d.TCI) & (d.TRC > d.TCO) & (d.TCA > 0)
    d['included'] = valid
    d['exclusion_reason'] = ''
    d.loc[~np.isfinite(d[required]).all(axis=1), 'exclusion_reason'] += 'Missing or nonfinite core value. '
    d.loc[d.TRC <= d.TCO, 'exclusion_reason'] += 'Saturation temperature not above leaving-water temperature. '
    d.loc[d.TCO <= d.TCI, 'exclusion_reason'] += 'Nonpositive condenser-water temperature rise. '
    d.loc[(d.FWC <= 0)|(d['Cond Tons'] <= 0)|(d['Evap Tons'] <= 0)|(d.kW <= 0)|(d.TCA <= 0), 'exclusion_reason'] += 'Nonpositive operating quantity. '
    d.to_csv(output/'source_steady_states_with_flags.csv', index=False)
    rejected = d.loc[~valid].copy()
    rejected.to_csv(output/'excluded_records.csv', index=False)
    d = d.loc[valid].copy()
    if d.duplicated(['source_file', 'Test']).any():
        raise ValueError('Duplicate source-file/Test identifiers require investigation.')
    d['condition_id'] = d.Test.astype(int)
    for source, target in [('TCI', 'water_in_C'), ('TCO', 'water_out_C'),
                           ('TRC', 'refrigerant_sat_C'), ('TEO', 'evaporator_out_C')]:
        d[target] = (d[source]-32)*5/9
    d['flow_kg_s'] = d.FWC * GPM_TO_M3_S * RHO
    d['heat_duty_kW'] = d['Cond Tons'] * TON_TO_KW
    d['evaporator_load_kW'] = d['Evap Tons'] * TON_TO_KW
    d['recorded_approach_K'] = d.TCA*5/9  # a difference: never subtract 32
    d['physical_approach_K'] = d.refrigerant_sat_C-d.water_out_C
    d['water_delta_K'] = d.water_out_C-d.water_in_C
    d['water_duty_kW'] = d.flow_kg_s*CP*d.water_delta_K
    d['water_duty_discrepancy_pct'] = 100*(d.water_duty_kW/d.heat_duty_kW-1)
    d['q_upper_kW'] = d.flow_kg_s*CP*(d.refrigerant_sat_C-d.water_in_C)
    # All epsilon/UA calculations use the same water-side duty for internal consistency.
    d['effectiveness'] = d.water_duty_kW/d.q_upper_kW
    d['ideal_headroom_kW'] = d.q_upper_kW-d.water_duty_kW
    d['ideal_min_flow_kg_s'] = d.water_duty_kW/(CP*(d.refrigerant_sat_C-d.water_in_C))
    hot_delta = d.refrigerant_sat_C-d.water_in_C
    cold_delta = d.physical_approach_K
    d['lmtd_K'] = (hot_delta-cold_delta)/np.log(hot_delta/cold_delta)
    d['effective_UA_kW_K'] = d.water_duty_kW/d.lmtd_K
    d['approach_sensor_difference_K'] = d.recorded_approach_K-d.physical_approach_K
    d.to_csv(output/'condenser_analysis.csv', index=False)
    assert ((d.effectiveness > 0) & (d.effectiveness < 1)).all()
    # Cross-run Test groups have a common test sequence, but measured loads/setpoints vary.
    ranges = d.groupby('condition_id').agg(
        inlet_span_K=('water_in_C', lambda v: v.max()-v.min()),
        evaporator_out_span_K=('evaporator_out_C', lambda v: v.max()-v.min()),
        load_span_kW=('evaporator_load_kW', lambda v: v.max()-v.min()))
    ranges.to_csv(output/'condition_group_variation.csv')
    audit = {'archive_sha256': None, 'selected_workbooks': len(selections()), 'source_rows': len(raw),
             'retained_rows': len(d), 'excluded_rows': len(rejected), 'condition_groups': d.condition_id.nunique(),
             'max_abs_water_duty_discrepancy_pct': float(d.water_duty_discrepancy_pct.abs().max()),
             'median_abs_water_duty_discrepancy_pct': float(d.water_duty_discrepancy_pct.abs().median()),
             'max_abs_approach_sensor_difference_K': float(d.approach_sensor_difference_K.abs().max()),
             'median_abs_recorded_heat_balance_pct': float(d['Heat Balance%'].abs().median()),
             'rows_abs_recorded_heat_balance_over_10pct': int((d['Heat Balance%'].abs()>10).sum()),
             'effectiveness_min': float(d.effectiveness.min()), 'effectiveness_max': float(d.effectiveness.max()),
             'observed_saturation_or_failure': False,
             'heat_duty_range_kW': [float(d.heat_duty_kW.min()), float(d.heat_duty_kW.max())],
             'physical_approach_range_K': [float(d.physical_approach_K.min()), float(d.physical_approach_K.max())],
             'units': {'temperatures': 'F to C: (F-32)*5/9', 'temperature_differences': 'F to K: F*5/9',
                       'flow': 'US gpm to kg/s using 997 kg/m3', 'heat_rate': 'refrigeration ton to kW: 3.5168528421'},
             'caveat': 'Water-side heat duty is calculated from the same flow and temperatures as the source Cond Tons. Their agreement is a unit check, not independent experimental validation.'}
    return d, audit


def score(y, pred):
    return {'MAE_K': float(mean_absolute_error(y, pred)),
            'RMSE_K': float(np.sqrt(mean_squared_error(y, pred))), 'R2': float(r2_score(y, pred))}


def train(d, output, models):
    X, y, groups = d[FEATURES], d[TARGET], d.condition_id
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED).split(X, y, groups))
    assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[te]))
    estimators = {'Mean baseline': DummyRegressor(),
                  'Ridge baseline': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                  'Random Forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                        random_state=SEED, n_jobs=-1)}
    results, cvrows = [], []
    fitted = {}
    for name, estimator in estimators.items():
        cvpred = np.full(len(tr), np.nan)
        for fold, (a, b) in enumerate(GroupKFold(5).split(X.iloc[tr], y.iloc[tr], groups.iloc[tr]), 1):
            model = clone(estimator).fit(X.iloc[tr].iloc[a], y.iloc[tr].iloc[a])
            cvpred[b] = model.predict(X.iloc[tr].iloc[b])
            cvrows.append({'model': name, 'fold': fold, **score(y.iloc[tr].iloc[b], cvpred[b])})
        model = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        fitted[name] = model
        results.append({'model': name, 'evaluation': 'training grouped 5-fold OOF', **score(y.iloc[tr], cvpred)})
        results.append({'model': name, 'evaluation': 'held-out operating conditions', **score(y.iloc[te], model.predict(X.iloc[te]))})
    pd.DataFrame(results).to_csv(output/'model_metrics.csv', index=False)
    pd.DataFrame(cvrows).to_csv(output/'cross_validation_folds.csv', index=False)
    rf = fitted['Random Forest']
    pred = rf.predict(X.iloc[te])
    predictions = d.iloc[te][['source_file', 'condition_id', 'fault_family', TARGET]].copy()
    predictions['predicted_approach_K'] = pred
    predictions['residual_K'] = predictions[TARGET]-pred
    predictions.to_csv(output/'heldout_predictions.csv', index=False)
    split = d[['source_file', 'condition_id']].copy()
    split['split'] = 'train'; split.iloc[te, split.columns.get_loc('split')] = 'test'
    split.to_csv(output/'split_assignments.csv', index=False)
    perm = permutation_importance(rf, X.iloc[te], y.iloc[te], scoring='neg_mean_absolute_error',
                                  n_repeats=40, random_state=SEED, n_jobs=1)
    ranking = pd.DataFrame({'feature': FEATURES, 'MAE_increase_K': perm.importances_mean,
                            'repeat_std_K': perm.importances_std,
                            'impurity_importance': rf.feature_importances_}).sort_values('MAE_increase_K', ascending=False)
    ranking.to_csv(output/'factor_ranking.csv', index=False)
    final = clone(estimators['Random Forest']).fit(X, y)
    bundle = {'model': final, 'features': FEATURES, 'target': TARGET, 'units': 'K',
              'feature_min': X.min().to_dict(), 'feature_max': X.max().to_dict(),
              'training_features': X, 'scope': 'RP-1043 normal, reduced condenser flow and tube blockage only',
              'sklearn_version': sklearn.__version__, 'validation': score(y.iloc[te], pred)}
    joblib.dump(bundle, models/'condenser_approach_model.joblib')
    joblib.dump({'model': rf, 'features': FEATURES, 'target': TARGET}, models/'evaluation_model_train_only.joblib')
    # Ensure serialized model reproduces the final model predictions.
    restored = joblib.load(models/'condenser_approach_model.joblib')
    np.testing.assert_allclose(restored['model'].predict(X.iloc[:5]), final.predict(X.iloc[:5]))
    metadata = {'seed': SEED, 'features': FEATURES, 'target': TARGET, 'train_rows': len(tr), 'test_rows': len(te),
                'train_groups': sorted(groups.iloc[tr].unique().tolist()),
                'test_groups': sorted(groups.iloc[te].unique().tolist()), 'rf_heldout': score(y.iloc[te], pred),
                'selection': 'RF specified in roadmap. Fixed parameters; no test-set tuning. Both baselines retained.',
                'deployment': 'Saved RF refit on all retained data after evaluation. Reported test metrics refer only to the train-only RF.',
                'limitations': 'Test indices identify nominal operating scenarios. Measured conditions differ across runs. Feature importance is predictive, not causal. Tube blockage fraction is known experimental metadata.'}
    # Sensitivity analysis for records with large source-reported energy imbalance.
    # Keep the original split and fixed model parameters. Do not tune on this result.
    balanced = d['Heat Balance%'].abs() <= 10
    tr_bal = [i for i in tr if balanced.iloc[i]]
    te_bal = [i for i in te if balanced.iloc[i]]
    sensitivity = clone(estimators['Random Forest']).fit(X.iloc[tr_bal], y.iloc[tr_bal])
    metadata['energy_balance_sensitivity'] = {'max_abs_heat_balance_pct': 10,
        'train_rows':len(tr_bal),'test_rows':len(te_bal),
        **score(y.iloc[te_bal], sensitivity.predict(X.iloc[te_bal]))}
    save_json(models/'model_metadata.json', metadata)
    return final, rf, metadata, ranking, predictions


def scenarios(d, model, output):
    base = d.loc[(d.source_file=='normal1.xls') & (d.condition_id==14)].iloc[0]
    x = base[FEATURES].astype(float)
    cases = [('Reference', 1., 0., 1.)]
    cases += [(f'Flow -{p}%', 1-p/100, 0., 1.) for p in [10, 20, 30, 40]]
    cases += [(f'Inlet +{p} K', 1., float(p), 1.) for p in [2, 4]]
    cases += [(f'Load +{p}%', 1., 0., 1+p/100) for p in [10, 20]]
    cases += [('Combined stress', .6, 4., 1.2)]
    rows = []
    train = d[FEATURES]
    scale = (train.max()-train.min()).replace(0, 1)
    norm = train/scale
    near = NearestNeighbors(n_neighbors=2).fit(norm)
    reference_dist = near.kneighbors(norm)[0][:,1]
    distance_limit = float(np.quantile(reference_dist, .95))
    for name, flow, temp, load in cases:
        row = x.copy();row.flow_kg_s *= flow; row.water_in_C += temp; row.evaporator_load_kW *= load
        frame = pd.DataFrame([row], columns=FEATURES)
        pred = float(model.predict(frame)[0])
        outside = [f for f in FEATURES if row[f] < train[f].min() or row[f] > train[f].max()]
        distance = float(near.kneighbors(frame/scale, n_neighbors=1)[0][0,0])
        weak = bool(outside or distance > distance_limit)
        rows.append({'scenario': name, **row.to_dict(), 'predicted_approach_K': pred,
                     'outside_feature_ranges': ','.join(outside), 'nearest_scaled_distance': distance,
                     'limited_local_support': weak})
    result = pd.DataFrame(rows)
    result['change_from_reference_K'] = result.predicted_approach_K-result.predicted_approach_K.iloc[0]
    result.to_csv(output/'what_if_predictions.csv', index=False)
    return result, {'reference_file': 'normal1.xls', 'reference_test': 14,
                    'distance_limit': distance_limit,
                    'support_rule': 'Outside marginal training range OR nearest scaled distance exceeds 95th percentile of leave-one-out nearest distances. This is a heuristic, not a confidence interval.',
                    'interpretation': 'Associational model scenarios. A larger condenser approach can indicate degraded heat transfer, while a smaller terminal approach can indicate greater ideal-limit utilization. Neither alone defines failure.'}


def matched(d, output):
    reference = d.loc[d.source_file=='normal1.xls'].set_index('condition_id')
    rows = []
    for name, group in d.groupby('source_file', sort=False):
        ref = d.loc[d.source_file=='normal cf.xls'].set_index('condition_id') if name.startswith('cf') else reference
        g = group.set_index('condition_id')
        common = g.index.intersection(ref.index)
        rows.append({'source_file': name, 'family': g.fault_family.iloc[0], 'conditions': len(common),
                     'nominal_reduction': float(g.nominal_flow_reduction.iloc[0]),
                     'blocked_fraction': float(g.blocked_tube_fraction.iloc[0]),
                     'mean_approach_K': float(g.recorded_approach_K.mean()),
                     'mean_effectiveness': float(g.effectiveness.mean()),
                     'mean_UA_kW_K': float(g.effective_UA_kW_K.mean()),
                     'matched_approach_change_K': float((g.loc[common,TARGET]-ref.loc[common,TARGET]).mean()),
                     'matched_COP_change_pct': float(((g.loc[common,'COP']/ref.loc[common,'COP']-1)*100).mean()),
                     'median_duty_change_pct': float(((g.loc[common,'heat_duty_kW']/ref.loc[common,'heat_duty_kW']-1)*100).median())})
    table = pd.DataFrame(rows);table.to_csv(output/'matched_run_comparisons.csv', index=False)
    return table


def lab(output):
    # Values transcribed from the supplied report at the user's direction.
    d = pd.DataFrame({'condition': ['Baseline', 'Surge'], 'water_in_C': [27.2,27.8],
                      'water_out_C': [34.5,66.8], 'hot_supply_C': [50.,70.],
                      'condensate_g': [120.,285.], 'duration_s': [600.,600.],
                      'reported_heater_W': [13.8,60.], 'assumed_flow_kg_s': [.05,.05]})
    d['water_duty_kW'] = d.assumed_flow_kg_s*CP*(d.water_out_C-d.water_in_C)
    d['mass_per_power_g_W'] = d.condensate_g/d.reported_heater_W
    d['mass_per_energy_kg_kWh'] = (d.condensate_g/1000)/(d.reported_heater_W/1000*d.duration_s/3600)
    d['water_to_heater_power_ratio'] = d.water_duty_kW*1000/d.reported_heater_W
    d.to_csv(output/'lab_report_values_and_checks.csv', index=False)
    return d


def recommendations(d, comparisons, output):
    flow40 = comparisons.loc[comparisons.source_file=='fwc40.xls'].iloc[0]
    block45 = comparisons.loc[comparisons.source_file=='cf45.xls'].iloc[0]
    rows = [
        {'finding':f'40% nominal flow reduction: mean matched COP change {flow40.matched_COP_change_pct:.2f}%.',
         'suggestion':'Restore design flow where cooling performance is constrained. Investigate pump or valve faults.',
         'expected_benefit':'Recover performance toward the matched benchmark. The measured change is not a guaranteed field saving.'},
        {'finding':f'74 of 164 tubes blocked: mean matched approach increase {block45.matched_approach_change_K:.3f} K.',
         'suggestion':'Inspect condenser tubes and remove blockage. Assess actual scale separately.',
         'expected_benefit':'Reduce thermal resistance. The experiment represents tube blockage rather than a measured scale layer.'},
        {'finding':'All retained records have positive terminal temperature approach and Q below the ideal bound.',
         'suggestion':'Use a supervised load sweep to establish an apparatus-specific allowable pressure or performance limit.',
         'expected_benefit':'Establish a defensible operating threshold and test whether variable-speed flow control is necessary.'},
        {'finding':'Load is the strongest predictor of recorded approach in the selected data.',
         'suggestion':'Monitor approach against expected load and inlet-water temperature, rather than one universal temperature.',
         'expected_benefit':'Distinguish expected load response from departures that merit investigation.'},
        {'finding':'Pump shaft power approximately follows speed cubed under similar-system assumptions.',
         'suggestion':'Evaluate variable-speed operation jointly with minimum flow, compressor power, pump curves and tower conditions.',
         'expected_benefit':'At 90% or 80% speed, ideal pump-only savings are 27.1% or 48.8%. Total plant savings remain undetermined.'}
    ]
    pd.DataFrame(rows).to_csv(output/'recommendations.csv', index=False)
    # Illustrative screening limit. 0.85 is an assumed planning limit, not an ASHRAE requirement.
    t=d[['source_file','condition_id','flow_kg_s','water_duty_kW','q_upper_kW','effectiveness','ideal_min_flow_kg_s']].copy()
    t['assumed_effectiveness_screen']=.85
    t['flow_for_085_screen_kg_s']=t.ideal_min_flow_kg_s/.85
    t['exceeds_assumed_screen']=t.effectiveness>.85
    t.to_csv(output/'illustrative_flow_screen.csv', index=False)
    return rows


def figures(d, comparisons, ranking, predictions, cases, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 140, 'savefig.dpi': 200, 'axes.titleweight': 'bold'})
    def finish(fig, name):
        fig.tight_layout();fig.savefig(output/name, bbox_inches='tight');plt.close(fig)
    fig, ax = plt.subplots(1,2,figsize=(11,4.1))
    for family,g in d.groupby('fault_family'):
        ax[0].scatter(g.flow_kg_s,g.heat_duty_kW,s=15,alpha=.65,c=COLORS[family],label=family.replace('_',' '))
    ax[0].set(xlabel='Cooling-water flow (kg/s)',ylabel='Condenser heat rejection (kW)',title='Operating envelope')
    ax[0].legend(fontsize=8)
    groups = [d.loc[d.fault_family==f,TARGET] for f in COLORS]
    ax[1].boxplot(groups,tick_labels=[f.replace('_','\n') for f in COLORS],showfliers=True)
    ax[1].set(ylabel='Recorded condenser approach (K)', title='Measured approach by test family')
    finish(fig,'01_operating_data.png')
    fig,ax=plt.subplots(1,2,figsize=(11,4.1))
    for family,g in d.groupby('fault_family'):
        ax[0].scatter(g.q_upper_kW,g.water_duty_kW,s=16,alpha=.6,c=COLORS[family],label=family.replace('_',' '))
    lim=d.q_upper_kW.max()*1.05;ax[0].plot([0,lim],[0,lim],'k--',label='Ideal upper bound')
    ax[0].set(xlabel='Ideal bound at measured saturation temperature (kW)',ylabel='Actual water-side duty (kW)',title='Thermal bound, not a measured failure threshold')
    ax[0].legend(fontsize=7)
    f=comparisons.loc[comparisons.family=='reduced_flow'].sort_values('nominal_reduction')
    ax[1].plot([0]+(f.nominal_reduction*100).tolist(),[d.loc[d.source_file=='normal1.xls','effectiveness'].mean()]+f.mean_effectiveness.tolist(),'o-',color='#256c85')
    ax[1].set(xlabel='Nominal condenser-flow reduction (%)',ylabel='Mean effectiveness Q / Q upper',title='Ideal-limit utilization across flow tests')
    finish(fig,'02_thermal_limit.png')
    fig,ax=plt.subplots(figsize=(8.5,4))
    q=ranking.iloc[::-1];ax.barh(q.feature,q.MAE_increase_K,xerr=q.repeat_std_K,color='#256c85')
    ax.set(xlabel='Increase in held-out MAE after permutation (K)',title='Predictive factor ranking')
    finish(fig,'03_factor_ranking.png')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for family,g in predictions.groupby('fault_family'):
        ax[0].scatter(g[TARGET],g.predicted_approach_K,s=20,c=COLORS[family],label=family.replace('_',' '))
    a,b=predictions[TARGET].min()-.2,predictions[TARGET].max()+.2
    ax[0].plot([a,b],[a,b],'k--');ax[0].set(xlabel='Measured approach (K)',ylabel='Predicted approach (K)',title='Held-out operating conditions');ax[0].legend(fontsize=7)
    ax[1].scatter(predictions.predicted_approach_K,predictions.residual_K,s=20,c='#256c85')
    ax[1].axhline(0,color='black',linestyle='--');ax[1].set(xlabel='Predicted approach (K)',ylabel='Measured minus predicted (K)',title='Prediction residuals')
    finish(fig,'04_model_validation.png')
    fig,ax=plt.subplots(figsize=(10,4.5))
    colors=['#c37529' if v else '#256c85' for v in cases.limited_local_support]
    ax.barh(cases.scenario.iloc[::-1],cases.predicted_approach_K.iloc[::-1],color=colors[::-1])
    ax.set(xlabel='Predicted approach (K)',title='Model scenarios at reference operating point')
    ax.text(.99,.02,'Orange: limited local data support',transform=ax.transAxes,ha='right',fontsize=8)
    finish(fig,'05_what_if.png')
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for family,reference in [('reduced_flow','normal1.xls'),('tube_blockage','normal cf.xls')]:
        g=comparisons.loc[comparisons.family==family].copy()
        severity=g.nominal_reduction if family=='reduced_flow' else g.blocked_fraction
        order=np.argsort(severity)
        label=family.replace('_',' ')
        ax[0].plot((severity.iloc[order]*100),g.matched_approach_change_K.iloc[order],'o-',label=label,c=COLORS[family])
        ax[1].plot((severity.iloc[order]*100),g.matched_COP_change_pct.iloc[order],'o-',label=label,c=COLORS[family])
    ax[0].set(xlabel='Flow reduction or blocked tubes (%)',ylabel='Mean matched approach change (K)',title='Changes relative to corresponding benchmark')
    ax[1].set(xlabel='Flow reduction or blocked tubes (%)',ylabel='Mean matched COP change (%)',title='Chiller efficiency response')
    for a in ax:a.axhline(0,c='gray',ls='--',lw=.8);a.legend(fontsize=8)
    finish(fig,'06_matched_comparisons.png')


def main(root=ROOT):
    root=Path(root);out=root/'output';data=out/'data';plots=out/'figures';models=out/'models'
    for p in [data,plots,models]:p.mkdir(parents=True,exist_ok=True)
    archive=root/'ASHRAE RP-1043.zip'
    if not archive.is_file():raise FileNotFoundError(f'Required source archive not found: {archive}')
    raw=load_data(archive,data);d,audit=prepare(raw,data)
    audit['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
    save_json(data/'data_audit.json',audit)
    comparisons=matched(d,data);labdata=lab(data)
    recommendations(d,comparisons,data)
    final,evaluated,metadata,ranking,predictions=train(d,data,models)
    cases,scenario_meta=scenarios(d,final,data);save_json(data/'scenario_metadata.json',scenario_meta)
    figures(d,comparisons,ranking,predictions,cases,plots)
    save_json(out/'presentation_data.json',{
        'points':d[['fault_family','flow_kg_s','heat_duty_kW','q_upper_kW','water_duty_kW']].to_dict('records'),
        'comparisons':comparisons.to_dict('records'), 'ranking':ranking.to_dict('records'),
        'metrics':pd.read_csv(data/'model_metrics.csv').to_dict('records'),
        'scenarios':cases.to_dict('records')})
    save_json(out/'results_summary.json',{'audit':audit,'model':metadata,
              'top_predictive_factor':ranking.iloc[0].feature,
              'max_effectiveness_record':d.loc[d.effectiveness.idxmax(),['source_file','condition_id','water_duty_kW','q_upper_kW','physical_approach_K','effectiveness']].to_dict(),
              'scenario_reference':scenario_meta})
    print(json.dumps({'rows':len(d),'groups':d.condition_id.nunique(),'RF heldout':metadata['rf_heldout'],
                      'top factor':ranking.iloc[0].feature,'max effectiveness':audit['effectiveness_max']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    main(parser.parse_args().root)
