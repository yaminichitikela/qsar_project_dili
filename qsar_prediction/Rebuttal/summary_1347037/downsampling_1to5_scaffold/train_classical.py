"""Classical ML (RF / SVM / XGBoost) on Caspase-3/7 CHO-K1 (AID 1347037), 1:5 downsampled training set.
Usage: python train_classical.py <scaffold> <rf|svm|xgb>
Requires prepare_scaffold_1to5.py to have been run first.
Same hyperparameter grids, selection rule and metrics as the published
../modeling/wahab_jobs/train_classical_<protocol> scripts; one run per split unit, then
mean +/- std across units.
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
from protocols import from_argv, unit_dir, prob_name

P = from_argv('<rf|svm|xgb>')

SEED = 42

if len(sys.argv) != 2 or sys.argv[1] not in ('rf', 'svm', 'xgb'):
    print(f'Usage: python {sys.argv[0]} <scaffold> <rf|svm|xgb>')
    sys.exit(1)
MODEL = sys.argv[1]
MODEL_NAME = {'rf': 'Random Forest', 'svm': 'SVM', 'xgb': 'XGBoost'}[MODEL]

OUT_DIR = P['out_dir']
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


repeat_rows = []
per_repeat_test_metrics = {0.5: [], 0.8: []}

for rep in range(P['n']):
    rep_dir = unit_dir(P, rep)
    X_train = np.load(os.path.join(rep_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(rep_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(rep_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(rep_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(rep_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(rep_dir, 'y_test.npy'))

    best_auc, best_prob_val, best_prob_test, best_params = 0, None, None, {}
    for params in GRIDS[MODEL]:
        clf, prob_val, prob_test = fit_predict(params, X_train, y_train, X_val, X_test)
        auc = roc_auc_score(y_val, prob_val)
        if auc > best_auc:
            best_auc, best_prob_val, best_prob_test, best_params = auc, prob_val, prob_test, params

    print(f'{P["unit_col"]} {rep}: best val AUC={best_auc:.4f}  params={best_params}')

    # save this unit's winning test-set probs, needed by compute_consensus.py
    probs_dir = os.path.join(OUT_DIR, 'probs')
    os.makedirs(probs_dir, exist_ok=True)
    np.save(os.path.join(probs_dir, prob_name(P, MODEL, rep)), best_prob_test)

    for cpt in [0.5, 0.8]:
        row_val = {**evaluate(y_val, best_prob_val, MODEL_NAME, cpt), 'Split': 'Val', P['unit_col']: rep}
        row_test = {**evaluate(y_test, best_prob_test, MODEL_NAME, cpt), 'Split': 'Test', P['unit_col']: rep}
        repeat_rows.append(row_val)
        repeat_rows.append(row_test)
        per_repeat_test_metrics[cpt].append(row_test)

df_reps = pd.DataFrame(repeat_rows)[[P['unit_col'], 'Split'] + METRIC_COLS]
df_reps.to_csv(os.path.join(OUT_DIR, f'{MODEL}_per_unit.csv'), index=False)

# ── average across units ────────────────────────────────────────────────────────
summary_rows = []
for cpt in [0.5, 0.8]:
    test_df = pd.DataFrame(per_repeat_test_metrics[cpt])
    numeric_cols = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
    mean_row = test_df[numeric_cols].mean().round(4)
    std_row = test_df[numeric_cols].std().round(4)
    summary_rows.append({'Model': MODEL_NAME, 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
    summary_rows.append({'Model': MODEL_NAME, 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, f'{MODEL}_summary.csv'), index=False)
print(f'\n== Mean +/- std across {P["n"]} {P["unit"]}s (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/{MODEL}_per_unit.csv and {MODEL}_summary.csv')
