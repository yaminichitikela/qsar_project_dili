"""Combines the 5 Stage-2 fold results (train_stage2_fold.py 0-4) into the
official Proxy+DILI report: ensembles the 5 fine-tuned fold-models'
held-out-223 predictions, and also reports the single best-CV-fold model
alone for comparison. Same structure as aggregate_stage1_cv.py.

Requires all 5 train_stage2_fold.py runs to have finished first.

Usage: python aggregate_stage2_cv.py
Outputs (./results/): mvgnn_cv_stage2_results.csv, mvgnn_cv_stage2_ensemble_test_probs.npy
"""
import os

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS

FOLDS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'stage2_folds')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')


def main():
    fold_val_aucs, fold_test_probs = [], []
    y_test = None
    for fold in range(5):
        path = os.path.join(FOLDS_DIR, f'fold_{fold}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {path} -- run train_stage2_fold.py {fold} first.')
        npz = np.load(path)
        fold_val_aucs.append(float(npz['val_auc']))
        fold_test_probs.append(npz['test_probs'])
        y_test = npz['y_test']

    fold_val_aucs = np.array(fold_val_aucs)
    fold_test_probs = np.array(fold_test_probs)
    print(f'Stage-2 fold val AUCs: {fold_val_aucs.round(4).tolist()}  '
          f'mean={fold_val_aucs.mean():.4f} +/- {fold_val_aucs.std():.4f}')

    ensemble_prob = np.mean(fold_test_probs, axis=0)
    best_fold = int(np.argmax(fold_val_aucs))
    single_prob = fold_test_probs[best_fold]
    print(f'Best single fold by val AUC: fold {best_fold} (val AUC={fold_val_aucs[best_fold]:.4f})')

    rows = []
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_test, ensemble_prob, 'MV-GNN-CV Stage-2 (Proxy+DILI) Ensemble', cpt), 'Split': 'Test'})
        rows.append({**evaluate(y_test, single_prob, 'MV-GNN-CV Stage-2 (Proxy+DILI) BestFold', cpt), 'Split': 'Test'})

    df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
    df.to_csv(os.path.join(OUT_DIR, 'mvgnn_cv_stage2_results.csv'), index=False)
    np.save(os.path.join(OUT_DIR, 'mvgnn_cv_stage2_ensemble_test_probs.npy'), ensemble_prob)

    print('\n== Stage-2 (Proxy+DILI) final held-out-223 results ==')
    print(df.to_string(index=False))
    print(f'\nSaved -> {OUT_DIR}/mvgnn_cv_stage2_results.csv, mvgnn_cv_stage2_ensemble_test_probs.npy')


if __name__ == '__main__':
    main()
