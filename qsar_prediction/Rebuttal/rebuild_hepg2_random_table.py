"""
Rebuilds ONLY the HepG2 Caspase single-split table from the fresh, consistent
rerun (prep.slurm + all 10 train_*.py + consensus.slurm resubmitted together
on Wahab, 2026-10-01), replacing both the stale master-workbook numbers and
the previously-missing CI cell for that one table in rebuttal_results_tables.tex.

The old master-workbook sheet value for Random Forest ROC-AUC (0.8971) does
not match this fresh run (0.8755) -- confirmed twice independently (once via
a local regeneration attempt, once via this real Wahab rerun) -- so the old
sheet is stale relative to the current cleaned_data file, not this run being
wrong. This script reads the fresh results/*_results.csv directly instead of
the master workbook for this one table.

Usage: python rebuild_hepg2_random_table.py
"""
import os
import numpy as np
import pandas as pd

from table_helpers import (
    BASE, bootstrap_ci_metric, build_table, NUMERIC_COLS, load_npy, MODEL_DISPLAY,
)

RESULTS_DIR = os.path.join(BASE, 'summary_2_1437034/modeling/wahab_jobs/results')
Y_TEST = load_npy(os.path.join(BASE, 'summary_2_1437034/modeling/wahab_jobs/data/y_test.npy'))

MODEL_FILES = {
    'Random Forest': 'rf', 'SVM': 'svm', 'XGBoost': 'xgb',
    'GCN': 'gcn', 'GAT': 'gat', 'GIN': 'gin', 'SAGE': 'sage',
    'Graphormer': 'graphormer', 'GraphGPS': 'graphgps',
}
CLASSICAL = ['Random Forest', 'SVM', 'XGBoost']
GNN = ['GCN', 'GAT', 'GIN', 'SAGE']
GT = ['Graphormer', 'GraphGPS']
CONSENSUS_GROUPS = {
    'Consensus-Classical': CLASSICAL, 'Consensus-GNN': GNN, 'Consensus-GT': GT,
    'Consensus': CLASSICAL + GNN + GT,
}

print('Loading fresh per-model results...')
rows_by_model = {}
probs = {}
for disp, fname in MODEL_FILES.items():
    df = pd.read_csv(os.path.join(RESULTS_DIR, f'{fname}_results.csv'))
    test = df[df['Split'] == 'Test']
    rows_by_model[disp] = {}
    for _, r in test.iterrows():
        rows_by_model[disp][r['CPT']] = {c: r[c] for c in NUMERIC_COLS}
    probs[disp] = load_npy(os.path.join(RESULTS_DIR, f'{fname}_test_probs.npy'))

# Build Consensus groups fresh from the mean of member probabilities, then
# evaluate with the project's standard evaluate() so CPT 0.5/0.8 rows match
# the same convention as every other table.
import sys
sys.path.insert(0, os.path.join(BASE, 'summary_2_1437034/modeling/wahab_jobs'))
from eval_utils import evaluate

for group_name, members in CONSENSUS_GROUPS.items():
    cons_prob = np.mean([probs[m] for m in members], axis=0)
    probs[group_name] = cons_prob
    rows_by_model[group_name] = {}
    for cpt in (0.5, 0.8):
        rows_by_model[group_name][cpt] = evaluate(Y_TEST, cons_prob, group_name, cpt)

print('Computing bootstrap CI...')
roc_ci_map, pr_ci_map = {}, {}
for model, p in probs.items():
    roc_ci_map[model] = bootstrap_ci_metric(Y_TEST, p, 'ROC-AUC')
    pr_ci_map[model] = bootstrap_ci_metric(Y_TEST, p, 'PR-AUC')
    print(f'  {model}: ROC-AUC={rows_by_model[model][0.5]["ROC-AUC"]:.4f}  CI={roc_ci_map[model]}')

new_table = build_table(
    'Performance on the Caspase-3/7 HepG2 assay (AID 654+1347034) -- single 80/10/10 split, 1:1 downsampled train '
    '(regenerated run, confirmed self-consistent)',
    'tab:hepg2_random', None, None, rows_by_model, roc_ci_map, pr_ci_map)

# Splice into the existing .tex file, replacing the old HepG2-random table block.
tex_path = os.path.join(BASE, 'rebuttal_results_tables.tex')
with open(tex_path) as f:
    content = f.read()

start_marker = "\\caption{Performance on the Caspase-3/7 HepG2 assay (AID 654+1347034) -- single 80/10/10 split, 1:1 downsampled train}"
start_idx = content.index("\\begin{table*}[t]\n\\centering\n" + start_marker)
# find the end of this table block
end_marker = "\\end{table*}"
end_idx = content.index(end_marker, start_idx) + len(end_marker)

old_block = content[start_idx:end_idx]
content = content[:start_idx] + new_table.rstrip('\n') + content[end_idx:]

with open(tex_path, 'w') as f:
    f.write(content)

print(f'\nReplaced HepG2 single-split table in {tex_path}')
