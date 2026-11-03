"""Consensus (simple unweighted mean of predicted probabilities) across model
families, computed per random-split repeat then averaged across the 5 repeats
(AID 1347034 only, up-sampled train):
    Consensus-Classical = mean(RF, SVM, XGBoost)
    Consensus-GNN       = mean(GCN, GAT, GIN, SAGE)
    Consensus-GT        = mean(Graphormer, GraphGPS)        [corrected architectures]
    Consensus           = mean(all 9 models above)          [excludes MV-GNN]
    Consensus-MVGNN     = mean(all 9 models above + MV-GNN-cross)

MV-GNN-cross stays out of Consensus-GNN / Consensus, as in the other up-sampled
runs, and appears only in Consensus-MVGNN.

Requires all 10 training jobs to have finished (reads each repeat's
*_repeat<i>_test_probs.npy from ./results_random_repeat/probs/).

Usage: python compute_consensus.py
"""
import os

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS

N_REPEATS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_random_repeat')
PROBS_DIR = os.path.join(os.path.dirname(__file__), 'results_random_repeat', 'probs')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results_random_repeat')

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

for rep in range(N_REPEATS):
    y_test = np.load(os.path.join(DATA_DIR, f'repeat_{rep}', 'y_test.npy'))

    probs = {}
    for key in ALL_MODELS:
        path = os.path.join(PROBS_DIR, f'{key}_repeat{rep}_test_probs.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {path} -- run the repeat-{rep} training for {key} first.')
        probs[key] = np.load(path)

    for name, members in CONSENSUS_GROUPS.items():
        consensus_prob = np.mean([probs[k] for k in members], axis=0)
        for cpt in [0.5, 0.8]:
            row = {**evaluate(y_test, consensus_prob, name, cpt), 'Split': 'Test', 'Repeat': rep}
            repeat_rows.append(row)
            per_repeat_test_metrics[name][cpt].append(row)

    print(f'Repeat {rep}: consensus computed for {list(CONSENSUS_GROUPS)}')

df_reps = pd.DataFrame(repeat_rows)[['Repeat', 'Split'] + METRIC_COLS]
df_reps.to_csv(os.path.join(OUT_DIR, 'consensus_per_repeat.csv'), index=False)

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
print('\n== Consensus mean +/- std across 5 repeats (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/consensus_per_repeat.csv and consensus_summary.csv')
