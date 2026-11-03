"""Classical ML (RF, SVM, XGBoost) on the 5 repeated random splits.
Usage: python train_classical_random_repeat.py
Requires prepare_random_repeat_splits.py to have been run first (loads from
./data_random_repeat/).
Same grid-search recipe as train_rf.py / train_svm.py / train_xgb.py (grid
search on val ROC-AUC, evaluate on test), looped over 5 repeats -- same
pattern as train_gnn_random_repeat.py / train_gt_random_repeat.py.
"""
import warnings; warnings.filterwarnings('ignore')
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

from eval_utils import evaluate, METRIC_COLS

SEED = 42
N_REPEATS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_random_repeat')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results_random_repeat')
PROBS_DIR = os.path.join(OUT_DIR, 'probs')
os.makedirs(PROBS_DIR, exist_ok=True)

RF_GRID = ParameterGrid({'n_estimators': [300, 500], 'max_features': ['sqrt', 0.3],
                          'min_samples_leaf': [1, 3], 'max_depth': [None, 20]})
SVM_GRID = ParameterGrid({'C': [1, 10, 50], 'gamma': ['scale', 0.01, 0.001]})


def fit_best(model_fn, grid, X_train, y_train, X_val, y_val):
    best_auc, best_clf = -1, None
    for p in grid:
        clf = model_fn(p)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc, best_clf = auc, clf
    return best_clf, best_auc


MODEL_FNS = {
    'Random Forest': ('rf', lambda p: RandomForestClassifier(**p, class_weight='balanced', n_jobs=-1, random_state=SEED), RF_GRID),
    'SVM': ('svm', lambda p: SVC(**p, kernel='rbf', class_weight='balanced', probability=True, random_state=SEED), SVM_GRID),
}


def main():
    per_repeat_rows = {name: [] for name in list(MODEL_FNS) + ['XGBoost']}

    for r in range(N_REPEATS):
        rep_dir = os.path.join(DATA_DIR, f'repeat_{r}')
        X_train = np.load(os.path.join(rep_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(rep_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(rep_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(rep_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(rep_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(rep_dir, 'y_test.npy'))

        for name, (short, fn, grid) in MODEL_FNS.items():
            clf, _ = fit_best(fn, grid, X_train, y_train, X_val, y_val)
            prob_test = clf.predict_proba(X_test)[:, 1]
            np.save(os.path.join(PROBS_DIR, f'{short}_repeat{r}_test_probs.npy'), prob_test)
            for cpt in [0.5, 0.8]:
                row = {**evaluate(y_test, prob_test, name, cpt), 'Split': 'Test', 'Repeat': r}
                per_repeat_rows[name].append(row)

        sw = (y_train == 0).sum() / (y_train == 1).sum()
        xgb_grid = ParameterGrid({'n_estimators': [300, 500], 'max_depth': [4, 6, 8],
                                   'learning_rate': [0.05, 0.1], 'subsample': [0.8], 'colsample_bytree': [0.8]})
        xgb_clf, _ = fit_best(
            lambda p: XGBClassifier(**p, scale_pos_weight=sw, eval_metric='logloss',
                                     n_jobs=-1, random_state=SEED, verbosity=0),
            xgb_grid, X_train, y_train, X_val, y_val)
        prob_test = xgb_clf.predict_proba(X_test)[:, 1]
        np.save(os.path.join(PROBS_DIR, f'xgb_repeat{r}_test_probs.npy'), prob_test)
        for cpt in [0.5, 0.8]:
            row = {**evaluate(y_test, prob_test, 'XGBoost', cpt), 'Split': 'Test', 'Repeat': r}
            per_repeat_rows['XGBoost'].append(row)

        print(f'Repeat {r} done.')

    for name, fname in [('Random Forest', 'rf'), ('SVM', 'svm'), ('XGBoost', 'xgb')]:
        df_repeats = pd.DataFrame(per_repeat_rows[name])[['Repeat', 'Split'] + METRIC_COLS]
        df_repeats.to_csv(os.path.join(OUT_DIR, f'{fname}_random_repeat_per_repeat.csv'), index=False)

        summary_rows = []
        numeric_cols = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
        for cpt in [0.5, 0.8]:
            test_df = df_repeats[df_repeats['CPT'] == cpt]
            mean_row = test_df[numeric_cols].mean().round(4)
            std_row = test_df[numeric_cols].std().round(4)
            summary_rows.append({'Model': name, 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
            summary_rows.append({'Model': name, 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})
        pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, f'{fname}_random_repeat_summary.csv'), index=False)

    print(f'\nSaved -> {OUT_DIR}/{{rf,svm,xgb}}_random_repeat_{{per_repeat,summary}}.csv')


if __name__ == '__main__':
    main()
