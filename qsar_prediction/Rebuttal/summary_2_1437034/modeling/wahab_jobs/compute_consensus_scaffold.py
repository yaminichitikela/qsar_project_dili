"""Consensus (simple unweighted mean of predicted probabilities) across model
families, computed per scaffold repeat then averaged across repeats -- same
averaging convention as train_classical_scaffold.py / train_gnn_scaffold.py /
train_gt_scaffold.py / train_mvgnn_scaffold.py:
    Consensus-Classical = mean(RF, SVM, XGBoost)
    Consensus-GNN       = mean(GCN, GAT, GIN, SAGE)
    Consensus-GT        = mean(Graphormer, GraphGPS)
    Consensus           = mean(all 9 models above)                  [excludes MV-GNN]
    Consensus-MVGNN     = mean(all 9 models above + MV-GNN-cross)   [includes MV-GNN]

MV-GNN-cross is grouped with the GNN family architecturally (local GINEConv
message passing, same as GCN/GAT/GIN/SAGE -- not global self-attention like
Graphormer/GraphGPS). It's kept out of Consensus-GNN/Consensus so none of the
already-reported numbers change, and only appears in Consensus-MVGNN.

Requires all 10 individual train_classical_scaffold.py / train_gnn_scaffold.py /
train_gt_scaffold.py / train_mvgnn_scaffold.py runs to have finished first
(reads each repeat's winning *_repeat<i>_test_probs.npy from
./results_scaffold/probs/).

Usage: python compute_consensus_scaffold.py
"""
import os

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS

N_REPEATS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_scaffold')
PROBS_DIR = os.path.join(os.path.dirname(__file__), 'results_scaffold', 'probs')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results_scaffold')

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

df_repeats = pd.DataFrame(repeat_rows)[['Repeat', 'Split'] + METRIC_COLS]
df_repeats.to_csv(os.path.join(OUT_DIR, 'consensus_scaffold_per_repeat.csv'), index=False)

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
df_summary.to_csv(os.path.join(OUT_DIR, 'consensus_scaffold_summary.csv'), index=False)
print('\n== Consensus mean +/- std across 5 scaffold-split repeats (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/consensus_scaffold_per_repeat.csv and consensus_scaffold_summary.csv')
