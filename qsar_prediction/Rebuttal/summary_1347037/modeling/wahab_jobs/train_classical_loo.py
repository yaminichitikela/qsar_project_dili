"""Classical ML on the Leave-one-out 10-fold split.
Usage: python train_classical_loo.py <rf|svm|xgb>
Requires prepare_loo_folds.py to have been run first (loads from ./data_loo/).
Same hyperparameter grids as qsar_mito.ipynb / the 8:1:1 scripts -- just run
once per fold, then averaged, per Ma et al.: "We take the average of the
results from all folds as the final results too."
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from xgboost import XGBClassifier

from eval_utils import evaluate, METRIC_COLS

SEED = 42
N_FOLDS = 10

if len(sys.argv) != 2 or sys.argv[1] not in ('rf', 'svm', 'xgb'):
    print(f'Usage: python {sys.argv[0]} <rf|svm|xgb>')
    sys.exit(1)
MODEL = sys.argv[1]
MODEL_NAME = {'rf': 'Random Forest', 'svm': 'SVM', 'xgb': 'XGBoost'}[MODEL]

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_loo')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results_loo')
os.makedirs(OUT_DIR, exist_ok=True)

GRIDS = {
    'rf': ParameterGrid({
        'n_estimators': [300, 500], 'max_features': ['sqrt', 0.3],
        'min_samples_leaf': [1, 3], 'max_depth': [None, 20],
    }),
    'svm': ParameterGrid({'C': [1, 10, 50], 'gamma': ['scale', 0.01, 0.001]}),
    'xgb': ParameterGrid({
        'n_estimators': [300, 500], 'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1], 'subsample': [0.8], 'colsample_bytree': [0.8],
    }),
}


def fit_predict(params, X_train, y_train, X_val, X_test):
    if MODEL == 'rf':
        clf = RandomForestClassifier(**params, class_weight='balanced', n_jobs=-1, random_state=SEED)
    elif MODEL == 'svm':
        clf = SVC(**params, kernel='rbf', class_weight='balanced', probability=True, random_state=SEED)
    else:  # xgb
        sw = (y_train == 0).sum() / (y_train == 1).sum()
        clf = XGBClassifier(**params, scale_pos_weight=sw, eval_metric='logloss',
                             n_jobs=-1, random_state=SEED, verbosity=0)
    clf.fit(X_train, y_train)
    return clf, clf.predict_proba(X_val)[:, 1], clf.predict_proba(X_test)[:, 1]


fold_rows = []
per_fold_test_metrics = {0.5: [], 0.8: []}

for fold in range(N_FOLDS):
    fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
    X_train = np.load(os.path.join(fold_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(fold_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(fold_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(fold_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(fold_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(fold_dir, 'y_test.npy'))

    best_auc, best_prob_val, best_prob_test, best_params = 0, None, None, {}
    for params in GRIDS[MODEL]:
        clf, prob_val, prob_test = fit_predict(params, X_train, y_train, X_val, X_test)
        auc = roc_auc_score(y_val, prob_val)
        if auc > best_auc:
            best_auc, best_prob_val, best_prob_test, best_params = auc, prob_val, prob_test, params

    print(f'Fold {fold}: best val AUC={best_auc:.4f}  params={best_params}')

    # save this fold's winning test-set probs, needed by compute_consensus_loo.py
    probs_dir = os.path.join(OUT_DIR, 'probs')
    os.makedirs(probs_dir, exist_ok=True)
    np.save(os.path.join(probs_dir, f'{MODEL}_fold{fold}_test_probs.npy'), best_prob_test)

    for cpt in [0.5, 0.8]:
        row_val = {**evaluate(y_val, best_prob_val, MODEL_NAME, cpt), 'Split': 'Val', 'Fold': fold}
        row_test = {**evaluate(y_test, best_prob_test, MODEL_NAME, cpt), 'Split': 'Test', 'Fold': fold}
        fold_rows.append(row_val)
        fold_rows.append(row_test)
        per_fold_test_metrics[cpt].append(row_test)

df_folds = pd.DataFrame(fold_rows)[['Fold', 'Split'] + METRIC_COLS]
df_folds.to_csv(os.path.join(OUT_DIR, f'{MODEL}_loo_per_fold.csv'), index=False)

# ── average across folds (per Ma et al.) ────────────────────────────────────
summary_rows = []
for cpt in [0.5, 0.8]:
    test_df = pd.DataFrame(per_fold_test_metrics[cpt])
    numeric_cols = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
    mean_row = test_df[numeric_cols].mean().round(4)
    std_row = test_df[numeric_cols].std().round(4)
    summary_rows.append({'Model': MODEL_NAME, 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
    summary_rows.append({'Model': MODEL_NAME, 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, f'{MODEL}_loo_summary.csv'), index=False)
print('\n== Mean +/- std across 10 folds (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/{MODEL}_loo_per_fold.csv and {MODEL}_loo_summary.csv')
