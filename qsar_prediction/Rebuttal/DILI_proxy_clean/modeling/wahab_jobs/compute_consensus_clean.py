"""Combines all 10 models' held-out-223 predictions (on the InChIKey-cleaned
split) into consensus groups, per stage:
    Consensus-Classic = mean(RF, SVM, XGB)
    Consensus-GNN     = mean(GCN, GAT, GIN, SAGE)
    Consensus-GT      = mean(Graphormer, GraphGPS)
    Consensus         = mean(all 9 above)                    [excludes MV-GNN]
    Consensus-MVGNN   = mean(all 9 above + MV-GNN)            [includes MV-GNN]
Same "add, don't replace" grouping convention used throughout this project.

Requires all 10 models' stage1_test_probs.npy AND stage2_test_probs.npy to
already exist in ./results/ (train_classical_stage{1,2}.py, train_gnn_stage{1,2}.py
<GCN|GAT|GIN|SAGE>, train_gt_stage{1,2}.py <Graphormer|GraphGPS>, train_mvgnn_stage{1,2}.py).

Usage: python compute_consensus_clean.py
Outputs: ./results/proxy_clean_all_results.csv
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


def load_probs(key, stage):
    path = os.path.join(RESULTS_DIR, f'{key}_stage{stage}_test_probs.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing {path} -- run the Stage-{stage} training for {NAME_MAP[key]} first.')
    return np.load(path)


def main():
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    rows = []

    for stage, stage_label in [(1, 'Proxy-only'), (2, 'Proxy+DILI')]:
        probs = {k: load_probs(k, stage) for k in ALL_MODELS}
        print(f'Stage {stage} ({stage_label}): loaded {len(probs)} models\' probabilities.')

        for k in ALL_MODELS:
            for cpt in [0.5, 0.8]:
                rows.append(evaluate(y_test, probs[k], f'{NAME_MAP[k]} ({stage_label})', cpt))

        consensus_probs = {
            f'Consensus-Classic ({stage_label})': np.mean([probs[k] for k in CLASSICAL], axis=0),
            f'Consensus-GNN ({stage_label})': np.mean([probs[k] for k in GNN], axis=0),
            f'Consensus-GT ({stage_label})': np.mean([probs[k] for k in GT], axis=0),
            f'Consensus ({stage_label})': np.mean([probs[k] for k in CLASSICAL + GNN + GT], axis=0),
            f'Consensus-MVGNN ({stage_label})': np.mean([probs[k] for k in ALL_MODELS], axis=0),
        }
        for name, prob in consensus_probs.items():
            for cpt in [0.5, 0.8]:
                rows.append(evaluate(y_test, prob, name, cpt))

    df = pd.DataFrame(rows)[METRIC_COLS]
    out_path = os.path.join(RESULTS_DIR, 'proxy_clean_all_results.csv')
    df.to_csv(out_path, index=False)

    print('\n== All results (held-out-223, InChIKey-cleaned split, CPT=0.5) ==')
    print(df[df['CPT'] == 0.5].to_string(index=False))
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
