"""Xu et al. "Structure only" method, run on the 2048-bit ECFP4 used by the
rest of this work, for the DILI 10-fold CV / 1:1 downsampled protocol.

WHY
---
The published Structure-only row uses the 1024-bit fingerprint Xu et al.
specify, while every other model in our tables uses 2048 bits. This run keeps
Xu's method (RF-Gini bit selection + a stock RBF SVM) and changes only the
fingerprint, so the comparison is on one representation.

The 1024- and 2048-bit arrays are row-aligned (identical InChIKeys and labels,
verified), so the existing split indices apply unchanged and nothing needs
re-featurising.

ARMS
----
A        N = 180 fixed -- Xu's published cutoff, carried over as-is.
B        N chosen per fold on VALIDATION ROC-AUC from Xu's published grid
         {20,40,...,200}: their selection procedure, re-run at 2048 bits.
B_ext    N chosen the same way from an EXTENDED grid up to all 2048 bits,
         since 180 of 2048 is half the proportion 180 of 1024 was. This tests
         whether the method wants more bits at the longer fingerprint.

N is always selected on the validation fold, never on Held-223. The per-N
Held-223 curve is also written out, but as a diagnostic only -- it is not used
to choose N.

SPEED
-----
Selection fits use probability=False and rank by decision_function. ROC-AUC and
PR-AUC are rank-based, so this is identical to the calibrated ranking without
paying for SVC's internal 5-fold Platt calibration -- the same trick used in
DILI_tx2017/train_one_setting.py and DILI_split_protocols/trainer.py. Only the
chosen model is refit calibrated, because the CPT-0.5/0.8 metrics need real
probabilities.

Usage: python run_tx_2048.py [--quick]
Outputs (results/): per_fold.csv, n_curve.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
TX = os.path.join(REB, 'DILI_tx2017_splits')
DSP = os.path.join(REB, 'DILI_split_protocols')
sys.path.insert(0, TX)
from eval_utils import evaluate  # noqa: E402

SEED = 42
N_FOLDS = 10
CPTS = (0.5, 0.8)
XU_GRID = list(range(20, 201, 20))                                   # Xu's published grid
EXT_GRID = XU_GRID + [240, 280, 320, 360, 400, 500, 600, 800, 1000, 1500, 2048]
OUT = os.path.join(HERE, 'results')


def build_fold(stage, fold):
    """2048-bit arrays on the existing 1:1 downsampled rows for this fold."""
    Xc = np.load(f'{DSP}/data/X_clin.npy')
    yc = np.load(f'{DSP}/data/y_clin.npy').astype(int)
    d = f'{TX}/splits/cv/unit_{fold}'
    tr = np.load(f'{d}/idx_train.npy')
    va, it = np.load(f'{d}/idx_val.npy'), np.load(f'{d}/idx_itest.npy')
    assert not np.intersect1d(tr, np.union1d(va, it)).size, f'fold {fold}: train overlaps val/itest'
    n1 = int((yc[tr] == 1).sum())
    assert n1 * 2 == len(tr), f'fold {fold}: idx_train is not 1:1 ({n1} of {len(tr)})'

    X_tr, y_tr = Xc[tr], yc[tr]
    if stage == 2:                      # NCV-888 appended unresampled, as elsewhere
        X_tr = np.vstack([X_tr, np.load(f'{DSP}/data/X_ncv.npy')])
        y_tr = np.concatenate([y_tr, np.load(f'{DSP}/data/y_ncv.npy').astype(int)])
    return (X_tr, y_tr, Xc[va], yc[va],
            np.load(f'{DSP}/data/X_test.npy'),
            np.load(f'{DSP}/data/y_test.npy').astype(int))


def gini_rank(X, y):
    """All bits ranked by RF Gini importance -- Xu's selection method."""
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    return np.argsort(rf.feature_importances_)[::-1]


def fit_rank(X_tr, y_tr, Xs):
    """Stock Xu SVM, uncalibrated; returns decision scores for each X in Xs."""
    clf = SVC(kernel='rbf', probability=False, random_state=SEED).fit(X_tr, y_tr)
    return [clf.decision_function(X) for X in Xs]


def fit_proba(X_tr, y_tr, X_te):
    """Stock Xu SVM, calibrated -- needed for the CPT-based metrics."""
    clf = SVC(kernel='rbf', probability=True, random_state=SEED).fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def main():
    quick = '--quick' in sys.argv[1:]
    folds = [0] if quick else list(range(N_FOLDS))
    grid = [20, 180, 2048] if quick else EXT_GRID
    os.makedirs(OUT, exist_ok=True)
    curve, rows = [], []

    for stage in (1, 2):
        for fold in folds:
            X_tr, y_tr, X_va, y_va, X_te, y_te = build_fold(stage, fold)
            order = gini_rank(X_tr, y_tr)
            t0 = time.time()
            for n in grid:
                cols = np.sort(order[:n])
                s_va, s_te = fit_rank(X_tr[:, cols], y_tr, [X_va[:, cols], X_te[:, cols]])
                curve.append({'Stage': stage, 'Fold': fold, 'N': n,
                              'val_ROC-AUC': roc_auc_score(y_va, s_va),
                              'test_ROC-AUC': roc_auc_score(y_te, s_te),
                              'test_PR-AUC': average_precision_score(y_te, s_te)})
            c = pd.DataFrame([r for r in curve if r['Stage'] == stage and r['Fold'] == fold])
            # Arm A: Xu's published cutoff. Arms B / B_ext: best N on VALIDATION.
            picks = {'A_n180': 180,
                     'B_xu_grid': int(c[c.N.isin(XU_GRID)].sort_values('val_ROC-AUC').N.iloc[-1]),
                     'B_ext_grid': int(c.sort_values('val_ROC-AUC').N.iloc[-1])}
            for arm, n in picks.items():
                cols = np.sort(order[:n])
                p = fit_proba(X_tr[:, cols], y_tr, X_te[:, cols])
                for cpt in CPTS:
                    rows.append({'Arm': arm, 'Stage': stage, 'Fold': fold, 'N': n,
                                 **evaluate(y_te, p, 'Structure only', cpt)})
            print(f'stage {stage} fold {fold}: picks={picks}  ({time.time() - t0:.0f}s)', flush=True)

    tag = '_quick' if quick else ''
    pd.DataFrame(curve).to_csv(f'{OUT}/n_curve{tag}.csv', index=False)
    pd.DataFrame(rows).to_csv(f'{OUT}/per_fold{tag}.csv', index=False)
    print(f'\nwrote {OUT}/per_fold{tag}.csv and n_curve{tag}.csv')


if __name__ == '__main__':
    main()
