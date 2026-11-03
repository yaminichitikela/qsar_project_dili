"""Combines the 5 Stage-1 fold results for ONE GT architecture
(train_gt_stage1_fold.py 0-4) into the official Proxy-only report.
Same structure as aggregate_gnn_stage1_cv.py.

Usage: python aggregate_gt_stage1_cv.py <Graphormer|GraphGPS>
Outputs (./results/): gt_cv_stage1_<arch>_results.csv,
                       gt_cv_stage1_<arch>_ensemble_test_probs.npy
"""
import os
import sys

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS

ARCHS = ('Graphormer', 'GraphGPS')


def main(arch):
    folds_dir = os.path.join(os.path.dirname(__file__), 'results', 'gt_stage1_folds', arch.lower())
    out_dir = os.path.join(os.path.dirname(__file__), 'results')

    fold_val_aucs, fold_test_probs = [], []
    y_test = None
    for fold in range(5):
        path = os.path.join(folds_dir, f'fold_{fold}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {path} -- run train_gt_stage1_fold.py {arch} {fold} first.')
        npz = np.load(path)
        fold_val_aucs.append(float(npz['val_auc']))
        fold_test_probs.append(npz['test_probs'])
        y_test = npz['y_test']

    fold_val_aucs = np.array(fold_val_aucs)
    fold_test_probs = np.array(fold_test_probs)
    print(f'{arch} Stage-1 fold val AUCs: {fold_val_aucs.round(4).tolist()}  '
          f'mean={fold_val_aucs.mean():.4f} +/- {fold_val_aucs.std():.4f}')

    ensemble_prob = np.mean(fold_test_probs, axis=0)
    best_fold = int(np.argmax(fold_val_aucs))
    single_prob = fold_test_probs[best_fold]
    print(f'Best single fold by val AUC: fold {best_fold} (val AUC={fold_val_aucs[best_fold]:.4f})')

    rows = []
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_test, ensemble_prob, f'{arch}-CV Stage-1 (Proxy-only) Ensemble', cpt), 'Split': 'Test'})
        rows.append({**evaluate(y_test, single_prob, f'{arch}-CV Stage-1 (Proxy-only) BestFold', cpt), 'Split': 'Test'})

    df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
    df.to_csv(os.path.join(out_dir, f'gt_cv_stage1_{arch.lower()}_results.csv'), index=False)
    np.save(os.path.join(out_dir, f'gt_cv_stage1_{arch.lower()}_ensemble_test_probs.npy'), ensemble_prob)

    print(f'\n== {arch} Stage-1 (Proxy-only) final held-out-223 results ==')
    print(df.to_string(index=False))
    print(f'\nSaved -> {out_dir}/gt_cv_stage1_{arch.lower()}_results.csv')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ARCHS:
        print(f'Usage: python {sys.argv[0]} <{"|".join(ARCHS)}>')
        sys.exit(1)
    main(sys.argv[1])
