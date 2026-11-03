"""
CI_Summary (in QSAR_Rebuttal_All_Results.xlsx) has Nadeau-Bengio CIs for every
individual model and the full 9-model Consensus, but not for the three
sub-group consensus types (Consensus-Classical, Consensus-GNN, Consensus-GT).
This fills that gap using the exact same corrected_ci() method already used
project-wide (compute_ci_significance.py), computed fresh from each dataset's
per-fold/per-repeat consensus_*_per_fold.csv / consensus_*_per_repeat.csv.

Usage: python fill_missing_consensus_ci.py
Writes a small lookup used by build_rebuttal_latex_tables.py (missing_consensus_ci.json).
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(__file__)
ALPHA = 0.05
SUBGROUPS = ['Consensus-Classical', 'Consensus-GNN', 'Consensus-GT']


def approx_train_test_sizes(n_total, n_active):
    n_test = n_total / 10.0
    n_train = 2.0 * (0.8 * n_active)
    return n_train, n_test


def corrected_ci(values, n_train, n_test, alpha=ALPHA):
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    var = sd ** 2
    corrected_var_of_mean = (1.0 / n + n_test / n_train) * var
    se = float(np.sqrt(corrected_var_of_mean))
    if n > 1 and se > 0:
        tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        moe = tcrit * se
    else:
        moe = 0.0
    return mean - moe, mean + moe


def dataset_counts(cleaned_csv):
    df = pd.read_csv(cleaned_csv, usecols=['PUBCHEM_ACTIVITY_OUTCOME'])
    counts = df['PUBCHEM_ACTIVITY_OUTCOME'].value_counts()
    n_active = int(counts.get('Active', 0))
    n_inactive = int(counts.get('Inactive', 0))
    return n_active + n_inactive, n_active


SPECS = [
    # (output_key, cleaned_csv, per_fold_csv, unit_col)
    ('MMP|LOO', 'Summary_1_models/modeling/wahab_jobs/cleaned_data/AID_720637_cleaned.csv',
     'Summary_1_models/modeling/wahab_jobs/results_loo/consensus_loo_per_fold.csv', 'Fold'),
    ('MMP|Scaffold', 'Summary_1_models/modeling/wahab_jobs/cleaned_data/AID_720637_cleaned.csv',
     'Summary_1_models/modeling/wahab_jobs/results_scaffold/consensus_scaffold_per_repeat.csv', 'Repeat'),
    ('HepG2|LOO', 'summary_2_1437034/modeling/wahab_jobs/cleaned_data/AID_654_1347034_cleaned.csv',
     'summary_2_1437034/modeling/wahab_jobs/results_loo/consensus_loo_per_fold.csv', 'Fold'),
    ('HepG2|Scaffold', 'summary_2_1437034/modeling/wahab_jobs/cleaned_data/AID_654_1347034_cleaned.csv',
     'summary_2_1437034/modeling/wahab_jobs/results_scaffold/consensus_scaffold_per_repeat.csv', 'Repeat'),
    ('HepG2|Up-sampling LOO', 'summary_2_1437034/modeling/wahab_jobs/cleaned_data/AID_654_1347034_cleaned.csv',
     'summary_2_1437034/upsampling/results_loo/consensus_loo_per_fold.csv', 'Fold'),
    ('CHOK1|LOO', 'summary_1347037/modeling/wahab_jobs/cleaned_data/AID_1347037_cleaned.csv',
     'summary_1347037/modeling/wahab_jobs/results_loo/consensus_loo_per_fold.csv', 'Fold'),
    ('CHOK1|Scaffold', 'summary_1347037/modeling/wahab_jobs/cleaned_data/AID_1347037_cleaned.csv',
     'summary_1347037/modeling/wahab_jobs/results_scaffold/consensus_scaffold_per_repeat.csv', 'Repeat'),
    ('CHOK1|Up-sampling LOO', 'summary_1347037/modeling/wahab_jobs/cleaned_data/AID_1347037_cleaned.csv',
     'summary_1347037/upsampling/results_loo/consensus_loo_per_fold.csv', 'Fold'),
]

out = {}
for key, cleaned_csv, per_fold_csv, unit_col in SPECS:
    cleaned_path = os.path.join(BASE, cleaned_csv)
    per_fold_path = os.path.join(BASE, per_fold_csv)
    if not os.path.exists(cleaned_path) or not os.path.exists(per_fold_path):
        print(f'SKIP {key}: missing file(s)')
        continue
    n_total, n_active = dataset_counts(cleaned_path)
    n_train, n_test = approx_train_test_sizes(n_total, n_active)
    df = pd.read_csv(per_fold_path)
    out[key] = {}
    for sg in SUBGROUPS:
        sub = df[(df['Split'] == 'Test') & (df['CPT'] == 0.5) & (df['Model'] == sg)]
        if sub.empty:
            print(f'  {key} / {sg}: no rows found')
            continue
        out[key][sg] = {}
        for metric in ('ROC-AUC', 'PR-AUC'):
            vals = sub.sort_values(unit_col)[metric].to_numpy(dtype=float)
            lo, hi = corrected_ci(vals, n_train, n_test)
            out[key][sg][metric] = [round(lo, 4), round(hi, 4)]
            print(f'  {key} / {sg} / {metric}: n={len(vals)} mean={vals.mean():.4f} CI=[{lo:.4f}, {hi:.4f}]')

# CI_Summary is also missing RF/XGBoost/Graphormer/GraphGPS specifically for
# the "Up-sampling LOO" method (both caspase datasets) -- fill from each
# model's own per-fold CSV the same way.
INDIV_MODEL_FILE = {
    'Random Forest': 'rf', 'XGBoost': 'xgb', 'Graphormer': 'graphormer', 'GraphGPS': 'graphgps',
}
INDIV_SPECS = [
    ('HepG2|Up-sampling LOO', 'summary_2_1437034/modeling/wahab_jobs/cleaned_data/AID_654_1347034_cleaned.csv',
     'summary_2_1437034/upsampling/results_loo', 'Fold'),
    ('CHOK1|Up-sampling LOO', 'summary_1347037/modeling/wahab_jobs/cleaned_data/AID_1347037_cleaned.csv',
     'summary_1347037/upsampling/results_loo', 'Fold'),
]
for key, cleaned_csv, per_fold_dir, unit_col in INDIV_SPECS:
    cleaned_path = os.path.join(BASE, cleaned_csv)
    if not os.path.exists(cleaned_path):
        print(f'SKIP {key} (indiv models): missing cleaned csv')
        continue
    n_total, n_active = dataset_counts(cleaned_path)
    n_train, n_test = approx_train_test_sizes(n_total, n_active)
    out.setdefault(key, {})
    for model_disp, fname_prefix in INDIV_MODEL_FILE.items():
        per_fold_path = os.path.join(BASE, per_fold_dir, f'{fname_prefix}_loo_per_fold.csv')
        if not os.path.exists(per_fold_path):
            print(f'  {key} / {model_disp}: per-fold file missing')
            continue
        df = pd.read_csv(per_fold_path)
        sub = df[(df['Split'] == 'Test') & (df['CPT'] == 0.5)]
        if sub.empty:
            print(f'  {key} / {model_disp}: no Test/CPT=0.5 rows')
            continue
        out[key][model_disp] = {}
        for metric in ('ROC-AUC', 'PR-AUC'):
            vals = sub.sort_values(unit_col)[metric].to_numpy(dtype=float)
            lo, hi = corrected_ci(vals, n_train, n_test)
            out[key][model_disp][metric] = [round(lo, 4), round(hi, 4)]
            print(f'  {key} / {model_disp} / {metric}: n={len(vals)} mean={vals.mean():.4f} CI=[{lo:.4f}, {hi:.4f}]')

with open(os.path.join(BASE, 'missing_consensus_ci.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('\nSaved -> missing_consensus_ci.json')
