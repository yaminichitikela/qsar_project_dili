"""Classical models (RF, SVM, XGB) Stage-2 ("Proxy+DILI"): same HP-search
protocol as Stage-1, but trained on clin_train + NCV-888 concatenated, still
validated on clin_val, evaluated on held-out-223. Independent HP search (not a
fine-tune -- classical models don't have continuable weights), matching how
RF/XGB already work in proxy_to_dili_consistent copy 2.ipynb.

Requires prepare_proxy_data_clean.py to have run first.
Usage: python train_classical_stage2.py
Outputs (./results/): classical_stage2_results.csv, {rf,svm,xgb}_stage2_test_probs.npy
"""
import os
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from train_classical_stage1 import search_rf, search_xgb, search_svm
from eval_utils import evaluate, METRIC_COLS

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    X_tr = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_tr = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    X_ncv = np.load(os.path.join(DATA_DIR, 'X_ncv.npy'))
    y_ncv = np.load(os.path.join(DATA_DIR, 'y_ncv.npy'))
    X_va = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    y_va = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    X_te = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_te = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    X_aug = np.vstack([X_tr, X_ncv])
    y_aug = np.concatenate([y_tr, y_ncv])
    print(f'Stage-2 train (clin_train + NCV-888): {len(y_aug)}  val: {len(y_va)}  test: {len(y_te)}')

    rows = []
    for name, search_fn in [('RF', search_rf), ('XGB', search_xgb)]:
        clf, val_auc = search_fn(X_aug, y_aug, X_va, y_va)
        prob_test = clf.predict_proba(X_te)[:, 1]
        print(f'{name}: val AUC={val_auc:.4f}')
        for cpt in [0.5, 0.8]:
            rows.append({**evaluate(y_te, prob_test, f'{name} Stage-2 (Proxy+DILI)', cpt), 'Split': 'Test'})
        np.save(os.path.join(OUT_DIR, f'{name.lower()}_stage2_test_probs.npy'), prob_test)

    svm_clf, svm_val_auc, svm_best_params = search_svm(X_aug, y_aug, X_va, y_va)
    prob_test = svm_clf.predict_proba(X_te)[:, 1]
    print(f'SVM: val AUC={svm_val_auc:.4f}  best_params={svm_best_params}')
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_te, prob_test, 'SVM Stage-2 (Proxy+DILI)', cpt), 'Split': 'Test'})
    np.save(os.path.join(OUT_DIR, 'svm_stage2_test_probs.npy'), prob_test)

    df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
    df.to_csv(os.path.join(OUT_DIR, 'classical_stage2_results.csv'), index=False)
    print(df)
    print(f'Saved -> {OUT_DIR}/classical_stage2_results.csv')


if __name__ == '__main__':
    main()
