"""Consensus (unweighted mean of predicted probabilities) per split unit, then
mean +/- std across units, for one up-sampled MMP protocol:
    Consensus-Classical = mean(RF, SVM, XGBoost)
    Consensus-GNN       = mean(GCN, GAT, GIN, SAGE)
    Consensus-GT        = mean(Graphormer, GraphGPS)        [corrected architectures]
    Consensus           = mean(all 9 models above)          [excludes MV-GNN]
    Consensus-MVGNN     = mean(all 9 models above + MV-GNN-cross)

Usage: python compute_consensus.py <random_repeat|loo|scaffold>
"""
import os

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS
from protocols import from_argv, unit_dir, prob_name

P = from_argv('')

PROBS_DIR = os.path.join(P['out_dir'], 'probs')
OUT_DIR = P['out_dir']

CLASSICAL = ['rf', 'svm', 'xgb']
GNN = ['gcn', 'gat', 'gin', 'sage']
GT = ['graphormer', 'graphgps']
MVGNN = ['mvgnn']
ALL_MODELS = CLASSICAL + GNN + GT + MVGNN

CONSENSUS_GROUPS = {
    'Consensus-Classical': CLASSICAL,
    'Consensus-GNN': GNN,
    'Consensus-GT': GT,
    'Consensus': CLASSICAL + GNN + GT,
    'Consensus-MVGNN': ALL_MODELS,
}

repeat_rows = []
per_repeat_test_metrics = {name: {0.5: [], 0.8: []} for name in CONSENSUS_GROUPS}

for rep in range(P['n']):
    y_test = np.load(os.path.join(unit_dir(P, rep), 'y_test.npy'))

    probs = {}
    for key in ALL_MODELS:
        path = os.path.join(PROBS_DIR, prob_name(P, key, rep))
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {path} -- run the {P["unit"]}-{rep} training for {key} first.')
        probs[key] = np.load(path)

    for name, members in CONSENSUS_GROUPS.items():
        consensus_prob = np.mean([probs[k] for k in members], axis=0)
        for cpt in [0.5, 0.8]:
            row = {**evaluate(y_test, consensus_prob, name, cpt), 'Split': 'Test', P['unit_col']: rep}
            repeat_rows.append(row)
            per_repeat_test_metrics[name][cpt].append(row)

    print(f'{P["unit_col"]} {rep}: consensus computed for {list(CONSENSUS_GROUPS)}')

df_reps = pd.DataFrame(repeat_rows)[[P['unit_col'], 'Split'] + METRIC_COLS]
df_reps.to_csv(os.path.join(OUT_DIR, 'consensus_per_unit.csv'), index=False)

summary_rows = []
numeric_cols = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
for name in CONSENSUS_GROUPS:
    for cpt in [0.5, 0.8]:
        test_df = pd.DataFrame(per_repeat_test_metrics[name][cpt])
        mean_row = test_df[numeric_cols].mean().round(4)
        std_row = test_df[numeric_cols].std().round(4)
        summary_rows.append({'Model': name, 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
        summary_rows.append({'Model': name, 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, 'consensus_summary.csv'), index=False)
print(f'\n== Consensus mean +/- std across {P["n"]} {P["unit"]}s (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/consensus_per_unit.csv and consensus_summary.csv')
