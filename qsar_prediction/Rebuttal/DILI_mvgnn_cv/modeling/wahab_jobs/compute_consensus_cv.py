"""Combines all 10 models' 5-fold-CV ENSEMBLE held-out-223 predictions into
consensus groups, per stage -- the CV analog of DILI_proxy_clean's
compute_consensus_clean.py. Each model's "prediction" here is its own
5-fold ensemble (mean of 5 fold-models), not a single-split prediction.
    Consensus-Classic = mean(RF, SVM, XGB)
    Consensus-GNN     = mean(GCN, GAT, GIN, SAGE)
    Consensus-GT      = mean(Graphormer, GraphGPS)
    Consensus         = mean(all 9 above)                    [excludes MV-GNN]
    Consensus-MVGNN   = mean(all 9 above + MV-GNN)            [includes MV-GNN]

Requires every model's aggregate_*_cv.py (classical: train_classical_cv.py
itself) to have already produced its stage1/stage2 ensemble_test_probs.npy.

Usage: python compute_consensus_cv.py
Outputs: ./results/mvgnn_cv_all_results.csv
"""
import os

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

CLASSICAL = ['rf', 'svm', 'xgb']
GNN = ['gcn', 'gat', 'gin', 'sage']
GT = ['graphormer', 'graphgps']
MVGNN = ['mvgnn']
ALL_MODELS = CLASSICAL + GNN + GT + MVGNN

NAME_MAP = {'rf': 'RF', 'svm': 'SVM', 'xgb': 'XGB', 'gcn': 'GCN', 'gat': 'GAT',
            'gin': 'GIN', 'sage': 'SAGE', 'graphormer': 'Graphormer',
            'graphgps': 'GraphGPS', 'mvgnn': 'MV-GNN'}


def probs_path(key, stage):
    if key in CLASSICAL:
        return os.path.join(RESULTS_DIR, f'classical_cv_stage{stage}_{key}_ensemble_test_probs.npy')
    if key in GNN:
        return os.path.join(RESULTS_DIR, f'gnn_cv_stage{stage}_{key}_ensemble_test_probs.npy')
    if key in GT:
        return os.path.join(RESULTS_DIR, f'gt_cv_stage{stage}_{key}_ensemble_test_probs.npy')
    return os.path.join(RESULTS_DIR, f'mvgnn_cv_stage{stage}_ensemble_test_probs.npy')


def load_probs(key, stage):
    path = probs_path(key, stage)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing {path} -- run the Stage-{stage} CV pipeline for {NAME_MAP[key]} first.')
    return np.load(path)


def main():
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    rows = []

    for stage, stage_label in [(1, 'Proxy-only'), (2, 'Proxy+DILI')]:
        probs = {k: load_probs(k, stage) for k in ALL_MODELS}
        print(f'Stage {stage} ({stage_label}): loaded {len(probs)} models\' CV-ensemble probabilities.')

        for k in ALL_MODELS:
            for cpt in [0.5, 0.8]:
                rows.append(evaluate(y_test, probs[k], f'{NAME_MAP[k]}-CV ({stage_label})', cpt))

        consensus_probs = {
            f'Consensus-Classic-CV ({stage_label})': np.mean([probs[k] for k in CLASSICAL], axis=0),
            f'Consensus-GNN-CV ({stage_label})': np.mean([probs[k] for k in GNN], axis=0),
            f'Consensus-GT-CV ({stage_label})': np.mean([probs[k] for k in GT], axis=0),
            f'Consensus-CV ({stage_label})': np.mean([probs[k] for k in CLASSICAL + GNN + GT], axis=0),
            f'Consensus-MVGNN-CV ({stage_label})': np.mean([probs[k] for k in ALL_MODELS], axis=0),
        }
        for name, prob in consensus_probs.items():
            for cpt in [0.5, 0.8]:
                rows.append(evaluate(y_test, prob, name, cpt))

    df = pd.DataFrame(rows)[METRIC_COLS]
    out_path = os.path.join(RESULTS_DIR, 'mvgnn_cv_all_results.csv')
    df.to_csv(out_path, index=False)

    print('\n== All CV results (held-out-223, InChIKey-cleaned, 5-fold ensemble, CPT=0.5) ==')
    print(df[df['CPT'] == 0.5].to_string(index=False))
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
