"""Bootstrap intervals for the six metrics missing from Table S3 Part 7.

tab:chok1_up_random is a SINGLE 80/10/10 split, so there are no repeats to
take a spread across and the repeated-evaluation machinery in
add_all_metric_ci.py does not apply. Its ROC-AUC and PR-AUC intervals are a
percentile bootstrap of the test set; this adds the other six by the same
method, so the table stays internally consistent.

The method is not assumed. Reproducing the 28 published ROC-AUC and PR-AUC
intervals pins it exactly: 2000 resamples, numpy RandomState(42) restarted for
each (model, metric), resamples with fewer than two classes skipped, and the
reported half-width = (p97.5 - p2.5)/2. That configuration matches all 28, and
no other seed or half-width definition tried came close.

Graphormer, GraphGPS, Consensus-GT and Consensus (Full) read their predictions
from graph_transformer_corrected/; the copies under summary_1347037/upsampling/
predate that correction and do not reproduce the published table.

Usage: python bootstrap_chok1_up_random.py
Writes results/chok1_up_random_bootstrap_ci.csv (the table edit is separate).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'summary_1347037', 'upsampling')
C = os.path.join(HERE, 'graph_transformer_corrected', 'results', 'chok1', 'up_single', 'probs')
N_BOOT, SEED = 2000, 42
FREE = ('ROC-AUC', 'PR-AUC')
THRESH = ('Sensitivity', 'Specificity', 'CCR', 'MCC', 'Bal-Acc', 'F1')

BASE = {'Random Forest': 'rf', 'SVM': 'svm', 'XGBoost': 'xgb', 'GCN': 'gcn', 'GAT': 'gat',
        'GIN': 'gin', 'SAGE': 'sage', 'MV-GNN$^{\\text{cross}}$': 'mvgnn',
        'Consensus-Classical': 'consensus_classical', 'Consensus-GNN': 'consensus_gnn'}
CORR = {'Graphormer': 'Graphormer_0', 'GraphGPS': 'GraphGPS_0',
        'Consensus-GT': 'Consensus-GT_0', 'Consensus (Full)': 'Consensus_0'}


def probs(model):
    if model in CORR:
        return np.load(os.path.join(C, CORR[model] + '_test_probs.npy'))
    return np.load(os.path.join(D, 'results', BASE[model] + '_test_probs.npy'))


def metric(y, p, name, cpt):
    if name == 'ROC-AUC':
        return roc_auc_score(y, p)
    if name == 'PR-AUC':
        return average_precision_score(y, p)
    pred = (p >= cpt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return {'Sensitivity': sens * 100, 'Specificity': spec * 100,
            'CCR': (sens + spec) / 2 * 100,
            'MCC': matthews_corrcoef(y, pred),
            'Bal-Acc': balanced_accuracy_score(y, pred) * 100,
            'F1': f1_score(y, pred, zero_division=0)}[name]


def half(y, p, name, cpt):
    """(p97.5 - p2.5)/2 over N_BOOT resamples, rng restarted for this metric."""
    rng = np.random.RandomState(SEED)
    vals = []
    for _ in range(N_BOOT):
        idx = rng.randint(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], p[idx], name, cpt))
    v = np.asarray(vals, float)
    return (np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2


def main():
    y = np.load(os.path.join(D, 'data', 'y_test.npy')).astype(int)
    rows = []
    for model in list(BASE) + list(CORR):
        p = probs(model)
        for name in FREE:
            rows.append({'Model': model, 'CPT': None, 'Metric': name,
                         'Point': metric(y, p, name, 0.5),
                         'Half': half(y, p, name, 0.5)})
        for cpt in (0.5, 0.8):
            for name in THRESH:
                rows.append({'Model': model, 'CPT': cpt, 'Metric': name,
                             'Point': metric(y, p, name, cpt),
                             'Half': half(y, p, name, cpt)})
        print(f'  {model} done', flush=True)
    out = pd.DataFrame(rows)
    path = os.path.join(HERE, 'results', 'chok1_up_random_bootstrap_ci.csv')
    out.to_csv(path, index=False)
    print(f'{len(out)} rows -> {path}')


if __name__ == '__main__':
    main()
