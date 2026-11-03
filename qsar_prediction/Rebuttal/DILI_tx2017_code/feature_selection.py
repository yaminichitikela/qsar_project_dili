"""Feature selection, ported from Xu et al.'s ACTUAL GITHUB CODE cutoffs
(merged_feature_selection_pipeline.py), NOT the paper text -- this is the
deliberate code-fidelity variant of DILI_tx2017's paper-text version. The
paper text and their own public code disagree on all three cutoff-based
methods' ranges; DILI_tx2017 (paper-text version) uses what the paper says,
this folder (DILI_tx2017_code) uses what their code actually runs, since
their code is what was actually used to generate the numbers they published,
regardless of what the text describes:
  - Fisher's exact test: p-value cutoffs 0.01 -> 0.05, step 0.01 (5 cutoffs)
  - Single-feature AUC-ROC: cutoffs 0.51 -> 0.60, step 0.01 (10 cutoffs)
  - Random Forest importance (Gini): top-N in {100,200,...,1000} (10 cutoffs)
  - XGBoost importance (Gain): top-N in {100,200,...,1000} (10 cutoffs)
    XGBoost params match the paper's stated values (max_depth=3, lr=0.01,
    colsample_bytree=0.5) plus gamma=1/min_child_weight=1 from their GitHub
    code (not specified in the paper text). importance_type='gain' is set
    explicitly since XGBoost's feature_importances_ default is inconsistent
    across versions, and the paper requires Gain scores specifically.

Two-stage protocol: Stage-1 ("Proxy-only") selects features from clin_train
alone; Stage-2 ("Proxy+DILI") selects independently from clin_train +
NCV-888 combined -- matching how RF/XGB Stage-2 works in DILI_proxy_clean
(full independent re-search, not fine-tuning; classical models have no
weights to carry over between stages). Only the stage's own training data is
touched -- val/test are never used for feature selection, in either stage.

Requires prepare_features_clean.py to have been run first (loads from ./data/).

Usage: python feature_selection.py <1|2>
Outputs: ./results/stage<N>/feature_selection.json -- for every (method,
cutoff), the list of selected feature (bit) indices.
"""
import json
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

FISHER_CUTOFFS = [round(x, 2) for x in np.arange(0.01, 0.06, 0.01)]   # 0.01..0.05 (code)
AUC_CUTOFFS = [round(x, 2) for x in np.arange(0.51, 0.61, 0.01)]      # 0.51..0.60 (code)
TOPN_CUTOFFS = list(range(100, 1001, 100))                             # 100..1000 (code)

XGB_FS_PARAMS = dict(
    objective='binary:logistic', eval_metric='logloss',
    max_depth=3, learning_rate=0.01, colsample_bytree=0.5,
    gamma=1, min_child_weight=1, n_estimators=100,
    n_jobs=-1, random_state=SEED, verbosity=0,
    importance_type='gain',  # paper requires Gain scores explicitly -- XGBoost's
                              # default for feature_importances_ is inconsistently
                              # reported across versions/sources, so don't rely on it
)


def fisher_selection(X, y):
    """p-value per bit via Fisher's exact test on the 2x2 (bit, label) table."""
    n_feat = X.shape[1]
    pvalues = np.ones(n_feat)
    for j in range(n_feat):
        col = X[:, j]
        a = int(((col == 1) & (y == 1)).sum())
        b = int(((col == 1) & (y == 0)).sum())
        c = int(((col == 0) & (y == 1)).sum())
        d = int(((col == 0) & (y == 0)).sum())
        _, p = stats.fisher_exact([[a, b], [c, d]])
        pvalues[j] = p
    selections = {}
    for cutoff in FISHER_CUTOFFS:
        selections[cutoff] = np.where(pvalues <= cutoff)[0].tolist()
    return selections, pvalues


def auc_selection(X, y):
    """Per-bit AUC-ROC, using the bit value itself as the score."""
    n_feat = X.shape[1]
    aucs = np.zeros(n_feat)
    for j in range(n_feat):
        col = X[:, j]
        if col.min() == col.max():
            aucs[j] = 0.5
            continue
        auc = roc_auc_score(y, col)
        if auc < 0.5:
            auc = 1 - auc
        aucs[j] = auc
    selections = {}
    for cutoff in AUC_CUTOFFS:
        selections[cutoff] = np.where(aucs >= cutoff)[0].tolist()
    return selections, aucs


def rf_selection(X, y):
    model = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    model.fit(X, y)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    selections = {}
    for n in TOPN_CUTOFFS:
        selections[n] = order[:n].tolist()
    return selections, importances


def xgb_selection(X, y):
    model = XGBClassifier(**XGB_FS_PARAMS)
    model.fit(X, y)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    selections = {}
    for n in TOPN_CUTOFFS:
        selections[n] = order[:n].tolist()
    return selections, importances


def main(stage):
    OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', f'stage{stage}')
    os.makedirs(OUT_DIR, exist_ok=True)

    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    if stage == 2:
        X_ncv = np.load(os.path.join(DATA_DIR, 'X_ncv.npy'))
        y_ncv = np.load(os.path.join(DATA_DIR, 'y_ncv.npy'))
        X_train = np.vstack([X_train, X_ncv])
        y_train = np.concatenate([y_train, y_ncv])
        print(f'Stage-2 (Proxy+DILI): clin_train + NCV-888 combined')
    else:
        print(f'Stage-1 (Proxy-only): clin_train alone')
    print(f'X_train: {X_train.shape}  y_train: {y_train.shape}  positive rate: {y_train.mean():.3f}')

    all_selections = {}

    print('Running Fisher exact test feature selection...')
    fisher_sel, fisher_p = fisher_selection(X_train, y_train)
    all_selections['fisher'] = {str(k): v for k, v in fisher_sel.items()}
    for cutoff, feats in fisher_sel.items():
        print(f'  p<={cutoff}: {len(feats)} features selected')

    print('Running AUC-ROC feature selection...')
    auc_sel, auc_scores = auc_selection(X_train, y_train)
    all_selections['auc'] = {str(k): v for k, v in auc_sel.items()}
    for cutoff, feats in auc_sel.items():
        print(f'  auc>={cutoff}: {len(feats)} features selected')

    print('Running Random Forest importance feature selection...')
    rf_sel, rf_imp = rf_selection(X_train, y_train)
    all_selections['rf'] = {str(k): v for k, v in rf_sel.items()}
    for n, feats in rf_sel.items():
        print(f'  top-{n}: {len(feats)} features selected')

    print('Running XGBoost importance feature selection...')
    xgb_sel, xgb_imp = xgb_selection(X_train, y_train)
    all_selections['xgb'] = {str(k): v for k, v in xgb_sel.items()}
    for n, feats in xgb_sel.items():
        print(f'  top-{n}: {len(feats)} features selected')

    with open(os.path.join(OUT_DIR, 'feature_selection.json'), 'w') as f:
        json.dump(all_selections, f)

    total_settings = len(FISHER_CUTOFFS) + len(AUC_CUTOFFS) + 2 * len(TOPN_CUTOFFS)
    print(f'\nTotal feature-selection settings: {total_settings} '
          f'({len(FISHER_CUTOFFS)} fisher + {len(AUC_CUTOFFS)} auc + '
          f'{len(TOPN_CUTOFFS)} rf + {len(TOPN_CUTOFFS)} xgb)')
    print(f'Saved -> {OUT_DIR}/feature_selection.json')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('1', '2'):
        print(f'Usage: python {sys.argv[0]} <1|2>   (1=Proxy-only, 2=Proxy+DILI)')
        sys.exit(1)
    main(int(sys.argv[1]))
