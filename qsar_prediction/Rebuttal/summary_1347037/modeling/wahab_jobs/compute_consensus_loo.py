"""Consensus (simple unweighted mean of predicted probabilities) across model
families, computed per LOO fold then averaged across folds -- same averaging
convention as train_classical_loo.py / train_gnn_loo.py / train_gt_loo.py /
train_mvgnn_loo.py:
    Consensus-Classical = mean(RF, SVM, XGBoost)
    Consensus-GNN       = mean(GCN, GAT, GIN, SAGE)
    Consensus-GT        = mean(Graphormer, GraphGPS)
    Consensus           = mean(all 9 models above)                  [excludes MV-GNN]
    Consensus-MVGNN     = mean(all 9 models above + MV-GNN-cross)   [includes MV-GNN]

MV-GNN-cross is grouped with the GNN family architecturally (local GINEConv
message passing, same as GCN/GAT/GIN/SAGE -- not global self-attention like
Graphormer/GraphGPS). It's kept out of Consensus-GNN/Consensus so none of the
already-reported numbers change, and only appears in Consensus-MVGNN.

Requires all 10 individual train_classical_loo.py / train_gnn_loo.py /
train_gt_loo.py / train_mvgnn_loo.py runs to have finished first (reads each
fold's winning *_fold<i>_test_probs.npy from ./results_loo/probs/).

Usage: python compute_consensus_loo.py
"""
import os

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS

N_FOLDS = 10
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_loo')
PROBS_DIR = os.path.join(os.path.dirname(__file__), 'results_loo', 'probs')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results_loo')

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

fold_rows = []
per_fold_test_metrics = {name: {0.5: [], 0.8: []} for name in CONSENSUS_GROUPS}

for fold in range(N_FOLDS):
    y_test = np.load(os.path.join(DATA_DIR, f'fold_{fold}', 'y_test.npy'))

    probs = {}
    for key in ALL_MODELS:
        path = os.path.join(PROBS_DIR, f'{key}_fold{fold}_test_probs.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {path} -- run the fold-{fold} training for {key} first.')
        probs[key] = np.load(path)

    for name, members in CONSENSUS_GROUPS.items():
        consensus_prob = np.mean([probs[k] for k in members], axis=0)
        for cpt in [0.5, 0.8]:
            row = {**evaluate(y_test, consensus_prob, name, cpt), 'Split': 'Test', 'Fold': fold}
            fold_rows.append(row)
            per_fold_test_metrics[name][cpt].append(row)

    print(f'Fold {fold}: consensus computed for {list(CONSENSUS_GROUPS)}')

df_folds = pd.DataFrame(fold_rows)[['Fold', 'Split'] + METRIC_COLS]
df_folds.to_csv(os.path.join(OUT_DIR, 'consensus_loo_per_fold.csv'), index=False)

summary_rows = []
numeric_cols = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
for name in CONSENSUS_GROUPS:
    for cpt in [0.5, 0.8]:
        test_df = pd.DataFrame(per_fold_test_metrics[name][cpt])
        mean_row = test_df[numeric_cols].mean().round(4)
        std_row = test_df[numeric_cols].std().round(4)
        summary_rows.append({'Model': name, 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
        summary_rows.append({'Model': name, 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, 'consensus_loo_summary.csv'), index=False)
print('\n== Consensus mean +/- std across 10 folds (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/consensus_loo_per_fold.csv and consensus_loo_summary.csv')
