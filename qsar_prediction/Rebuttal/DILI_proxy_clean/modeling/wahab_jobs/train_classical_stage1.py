"""Classical models (RF, SVM, XGB) Stage-1 ("Proxy-only"): each does its own small
HP grid search (RF/XGB: 8 configs, SVM: 6 configs) on clin_train, picked by
clin_val ROC-AUC, then evaluated once on held-out-223. SVM search uses
decision_function() (ROC-AUC-invariant, avoids Platt-scaling cost); only the
winning SVM config gets refit with probability=True for calibrated probabilities.

Requires prepare_proxy_data_clean.py to have run first.
Usage: python train_classical_stage1.py
Outputs (./results/): classical_stage1_results.csv, {rf,svm,xgb}_stage1_test_probs.npy
"""
import os
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from classical_config import RF_GRID, XGB_GRID, SVM_GRID, make_rf, make_xgb, make_svm
from eval_utils import evaluate, METRIC_COLS

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def search_rf(X_tr, y_tr, X_va, y_va):
    best_auc, best_clf = 0, None
    for p in RF_GRID:
        clf = make_rf(p)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        if auc > best_auc:
            best_auc, best_clf = auc, clf
    return best_clf, best_auc


def search_xgb(X_tr, y_tr, X_va, y_va):
    best_auc, best_clf = 0, None
    for p in XGB_GRID:
        clf = make_xgb(p, y_tr)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        if auc > best_auc:
            best_auc, best_clf = auc, clf
    return best_clf, best_auc


def search_svm(X_tr, y_tr, X_va, y_va):
    best_auc, best_params = 0, None
    for p in SVM_GRID:
        clf = make_svm(p, calibrated=False)
        clf.fit(X_tr, y_tr)
        auc = roc_auc_score(y_va, clf.decision_function(X_va))
        if auc > best_auc:
            best_auc, best_params = auc, p
    final_clf = make_svm(best_params, calibrated=True)
    final_clf.fit(X_tr, y_tr)
    return final_clf, best_auc, best_params


def main():
    X_tr = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_tr = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_va = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_va = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_te = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_te = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    print(f'Stage-1 train/val/test: {len(y_tr)}/{len(y_va)}/{len(y_te)}')

    rows = []
    for name, search_fn in [('RF', search_rf), ('XGB', search_xgb)]:
        clf, val_auc = search_fn(X_tr, y_tr, X_va, y_va)
        prob_test = clf.predict_proba(X_te)[:, 1]
        print(f'{name}: val AUC={val_auc:.4f}')
        for cpt in [0.5, 0.8]:
            rows.append({**evaluate(y_te, prob_test, f'{name} Stage-1 (Proxy-only)', cpt), 'Split': 'Test'})
        np.save(os.path.join(OUT_DIR, f'{name.lower()}_stage1_test_probs.npy'), prob_test)

    svm_clf, svm_val_auc, svm_best_params = search_svm(X_tr, y_tr, X_va, y_va)
    prob_test = svm_clf.predict_proba(X_te)[:, 1]
    print(f'SVM: val AUC={svm_val_auc:.4f}  best_params={svm_best_params}')
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_te, prob_test, 'SVM Stage-1 (Proxy-only)', cpt), 'Split': 'Test'})
    np.save(os.path.join(OUT_DIR, 'svm_stage1_test_probs.npy'), prob_test)

    df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
    df.to_csv(os.path.join(OUT_DIR, 'classical_stage1_results.csv'), index=False)
    print(df)
    print(f'Saved -> {OUT_DIR}/classical_stage1_results.csv')


if __name__ == '__main__':
    main()
