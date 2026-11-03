"""Xu et al.'s FULL model-selection procedure, re-run on the 2048-bit ECFP4.

WHY THIS AND NOT THE FIXED PIPELINE
-----------------------------------
Xu et al.'s contribution is the search, not any one model: four feature-
selection methods with their cutoff grids, crossed with classifiers, the
winner chosen on validation ROC-AUC. "RF-Gini top-180 + SVM" is the *outcome*
of that search on their 1024-bit data, not the method itself.

Hard-coding that outcome and calling it their method freezes a result that was
selected for a different representation. On 2048 bits the search may well pick
a different selection method, cutoff, or classifier -- so here the procedure is
re-run and allowed to choose for itself. Only the input changes.

WHAT IS HELD FIXED (so the row is comparable to the rest of the table)
    - the 2048-bit ECFP4 used by every other model
    - the same split indices and the same 1:1 downsampled idx_train.npy
    - Held-223 scored once per fold, never used for any selection

WHAT IS XU'S, UNCHANGED
    - the four selection methods (Fisher exact, single-bit AUC, RF Gini,
      XGBoost Gain) and their cutoff semantics
    - the five classifiers at their original settings (library defaults;
      XGBoost at the paper's stated values)
    - selection on validation ROC-AUC

CUTOFF SCALING
    Fisher p and single-bit AUC are thresholds, so they carry over unchanged.
    The top-N grids are counts: {20..200} covered 2-20% of a 1024-bit vector,
    so they become {40..400} to cover the same proportion of 2048 bits.

SAMPLER
    Xu searches four samplers. Here the training rows are the 1:1 downsampled
    ones the other models in the table use, so balancing matches the rest of
    the table instead of being a fifth free axis.

SPEED
    Each fold-stage computes the four rankings once; the cutoffs are prefixes
    or thresholds on them. SVM is ranked by decision_function during the search
    (ROC-AUC is rank-invariant), and only the winner is refit calibrated.
    Results are checkpointed per fold-stage.

Usage: python run_full_search.py [--folds N] [--stages 1,2]
Output: results/full_search_cv.csv, results/full_search_cv_winners.csv
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
TX = os.path.join(REB, 'DILI_tx2017_splits')
DSP = os.path.join(REB, 'DILI_split_protocols')
sys.path.insert(0, TX)
from eval_utils import evaluate  # noqa: E402

SEED = 42
CPTS = (0.5, 0.8)
FISHER_CUTOFFS = [round(x, 2) for x in np.arange(0.01, 0.11, 0.01)]
AUC_CUTOFFS = [round(x, 2) for x in np.arange(0.51, 0.58, 0.01)]
TOPN_CUTOFFS = list(range(40, 401, 40))        # {20..200} of 1024 -> {40..400} of 2048

XGB_PARAMS = dict(objective='binary:logistic', eval_metric='logloss',
                  max_depth=3, learning_rate=0.01, colsample_bytree=0.5,
                  gamma=1, min_child_weight=1, n_estimators=100,
                  n_jobs=-1, random_state=SEED, verbosity=0)
CLASSIFIERS = {
    'NB': lambda: BernoulliNB(alpha=1.0),
    'SVM': lambda: SVC(kernel='rbf', probability=False, random_state=SEED),
    'RF': lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
    'NNET': lambda: MLPClassifier(random_state=SEED, max_iter=500),
    'XGBoost': lambda: XGBClassifier(**XGB_PARAMS),
}


def build(stage, fold):
    Xc = np.load(f'{DSP}/data/X_clin.npy')
    yc = np.load(f'{DSP}/data/y_clin.npy').astype(int)
    d = f'{TX}/splits/cv/unit_{fold}'
    tr, va = np.load(f'{d}/idx_train.npy'), np.load(f'{d}/idx_val.npy')
    assert not np.intersect1d(tr, va).size
    n1 = int((yc[tr] == 1).sum())
    assert n1 * 2 == len(tr), f'fold {fold}: idx_train not 1:1'
    X_tr, y_tr = Xc[tr], yc[tr]
    if stage == 2:
        X_tr = np.vstack([X_tr, np.load(f'{DSP}/data/X_ncv.npy')])
        y_tr = np.concatenate([y_tr, np.load(f'{DSP}/data/y_ncv.npy').astype(int)])
    return (X_tr, y_tr, Xc[va], yc[va],
            np.load(f'{DSP}/data/X_test.npy'), np.load(f'{DSP}/data/y_test.npy').astype(int))


def rankings(X, y):
    """The four selection scores, each computed once on the training rows."""
    pos, neg = y == 1, y == 0
    a = X[pos].sum(0)                      # bit on,  label 1
    b = X[neg].sum(0)                      # bit on,  label 0
    c = pos.sum() - a                      # bit off, label 1
    d = neg.sum() - b                      # bit off, label 0
    pvals = np.array([stats.fisher_exact([[int(a[j]), int(b[j])],
                                          [int(c[j]), int(d[j])]])[1]
                      for j in range(X.shape[1])])
    aucs = np.array([roc_auc_score(y, X[:, j]) if X[:, j].std() > 0 else 0.5
                     for j in range(X.shape[1])])
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(X, y)
    xgb = XGBClassifier(**{**XGB_PARAMS, 'importance_type': 'gain'}).fit(X, y)
    return {'fisher': pvals, 'auc': aucs,
            'rf': np.argsort(rf.feature_importances_)[::-1],
            'xgb': np.argsort(xgb.feature_importances_)[::-1]}


def settings(rk):
    """(method, cutoff, feature indices) -- Xu's 37-point grid at 2048 bits."""
    out = []
    for p in FISHER_CUTOFFS:
        out.append(('fisher', p, np.where(rk['fisher'] <= p)[0]))
    for t in AUC_CUTOFFS:
        out.append(('auc', t, np.where(np.maximum(rk['auc'], 1 - rk['auc']) >= t)[0]))
    for n in TOPN_CUTOFFS:
        out.append(('rf', n, np.sort(rk['rf'][:n])))
        out.append(('xgb', n, np.sort(rk['xgb'][:n])))
    return [(m, c, f) for m, c, f in out if len(f) > 0]


def val_auc(name, X_tr, y_tr, X_va, y_va):
    clf = CLASSIFIERS[name]().fit(X_tr, y_tr)
    s = clf.decision_function(X_va) if name == 'SVM' else clf.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, s)


def final_probs(name, X_tr, y_tr, X_te):
    """Winner refit with calibrated output, needed for the CPT metrics."""
    clf = (SVC(kernel='rbf', probability=True, random_state=SEED) if name == 'SVM'
           else CLASSIFIERS[name]())
    return clf.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


def main():
    args = sys.argv[1:]
    folds = range(int(args[args.index('--folds') + 1])) if '--folds' in args else range(10)
    stages = ([int(x) for x in args[args.index('--stages') + 1].split(',')]
              if '--stages' in args else [1, 2])
    os.makedirs(f'{HERE}/results', exist_ok=True)
    grid_rows, win_rows = [], []
    for stage in stages:
        for fold in folds:
            t0 = time.time()
            X_tr, y_tr, X_va, y_va, X_te, y_te = build(stage, fold)
            rk = rankings(X_tr, y_tr)
            best = None
            for method, cutoff, feat in settings(rk):
                for name in CLASSIFIERS:
                    v = val_auc(name, X_tr[:, feat], y_tr, X_va[:, feat], y_va)
                    grid_rows.append({'Stage': stage, 'Fold': fold, 'method': method,
                                      'cutoff': cutoff, 'n_feat': len(feat),
                                      'clf': name, 'val_ROC-AUC': v})
                    if best is None or v > best[0]:
                        best = (v, method, cutoff, feat, name)
            v, method, cutoff, feat, name = best
            p = final_probs(name, X_tr[:, feat], y_tr, X_te[:, feat])
            for cpt in CPTS:
                win_rows.append({'Stage': stage, 'Fold': fold, 'method': method,
                                 'cutoff': cutoff, 'n_feat': len(feat), 'clf': name,
                                 'val_ROC-AUC': round(v, 4),
                                 **evaluate(y_te, p, 'Structure only', cpt)})
            pd.DataFrame(grid_rows).to_csv(f'{HERE}/results/full_search_cv.csv', index=False)
            pd.DataFrame(win_rows).to_csv(f'{HERE}/results/full_search_cv_winners.csv', index=False)
            print(f'stage {stage} fold {fold}: winner={method}/{cutoff} '
                  f'({len(feat)} bits) + {name}  val={v:.4f}  '
                  f'test={win_rows[-1]["ROC-AUC"]:.4f}  ({time.time() - t0:.0f}s)', flush=True)
    print('\ndone')


if __name__ == '__main__':
    main()
