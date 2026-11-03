"""Consensus (simple unweighted mean of predicted probabilities) across model
families, same methodology as QSAR_Caspase_GraphTransformers.ipynb cells 29-30:
    Consensus-Classical = mean(RF, SVM, XGBoost)
    Consensus-GNN       = mean(GCN, GAT, GIN, SAGE)
    Consensus-GT        = mean(Graphormer, GraphGPS)
    Consensus           = mean(all 9 models above)                  [excludes MV-GNN]
    Consensus-MVGNN     = mean(all 9 models above + MV-GNN-cross)   [includes MV-GNN]

MV-GNN-cross is grouped with the GNN family architecturally (its two encoder
streams use GINEConv -- local, edge-restricted message passing, same as
GCN/GAT/GIN/SAGE -- not global self-attention like Graphormer/GraphGPS; its
only attention use is in the final readout/pooling step). It is kept out of
the original Consensus-GNN/Consensus groups so none of the already-reported
numbers change, and instead only appears in the new Consensus-MVGNN group.

Requires all 10 individual train_*.py / train_gnn.py / train_gt.py /
train_mvgnn.py runs to have finished first (reads each one's *_test_probs.npy
from ./results/).

Usage: python compute_consensus.py
"""
import os

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

CLASSICAL = ['rf', 'svm', 'xgb']
GNN = ['gcn', 'gat', 'gin', 'sage']
GT = ['graphormer', 'graphgps']
MVGNN = ['mvgnn']

NAME_MAP = {'rf': 'Random Forest', 'svm': 'SVM', 'xgb': 'XGBoost',
            'gcn': 'GCN', 'gat': 'GAT', 'gin': 'GIN', 'sage': 'SAGE',
            'graphormer': 'Graphormer', 'graphgps': 'GraphGPS',
            'mvgnn': 'MV-GNN-cross'}


TRAIN_SCRIPT = {**{k: f'train_{k}.py' for k in CLASSICAL},
                 **{k: f'train_gnn.py {NAME_MAP[k]}' for k in GNN},
                 **{k: f'train_gt.py {NAME_MAP[k]}' for k in GT},
                 **{k: 'train_mvgnn.py' for k in MVGNN}}


def load_probs(key):
    path = os.path.join(RESULTS_DIR, f'{key}_test_probs.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing {path} -- run {TRAIN_SCRIPT[key]} first.')
    return np.load(path)


y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

probs = {k: load_probs(k) for k in CLASSICAL + GNN + GT + MVGNN}
print('Loaded individual model probabilities:', {NAME_MAP[k]: probs[k].shape for k in probs})

consensus_probs = {
    'Consensus-Classical': np.mean([probs[k] for k in CLASSICAL], axis=0),
    'Consensus-GNN': np.mean([probs[k] for k in GNN], axis=0),
    'Consensus-GT': np.mean([probs[k] for k in GT], axis=0),
    'Consensus': np.mean([probs[k] for k in CLASSICAL + GNN + GT], axis=0),
    'Consensus-MVGNN': np.mean([probs[k] for k in CLASSICAL + GNN + GT + MVGNN], axis=0),
}

rows = []
for name, prob in consensus_probs.items():
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_test, prob, name, cpt), 'Split': 'Test'})
    np.save(os.path.join(RESULTS_DIR, f'{name.lower().replace("-", "_")}_test_probs.npy'), prob)

df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
df.to_csv(os.path.join(RESULTS_DIR, 'consensus_results.csv'), index=False)
print('\n== Consensus results (Test set) ==')
print(df)
print(f'\nSaved -> {RESULTS_DIR}/consensus_results.csv')
