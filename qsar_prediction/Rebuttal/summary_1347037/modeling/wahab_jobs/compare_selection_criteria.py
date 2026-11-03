"""Controlled test: does switching the model-selection criterion (which
hyperparameter config wins the grid search) from val ROC-AUC to val PR-AUC
or val F1@0.5 change test-set performance?

Trains every grid config ONCE per classical model, records val ROC-AUC,
val PR-AUC, val F1@0.5 for each, then picks a winning config separately
under each criterion and reports that winner's TEST metrics.

Usage: python compare_selection_criteria.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
from eval_utils import evaluate, METRIC_COLS
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

CRITERIA = ['ROC-AUC', 'PR-AUC', 'F1']


def run_grid(name, grid, make_clf):
    results = []  # list of (params, clf, val_roc, val_pr, val_f1)
    print(f'\n{name}: {len(grid)} configs...')
    for p in grid:
        clf = make_clf(p)
        clf.fit(X_train, y_train)
        prob_val = clf.predict_proba(X_val)[:, 1]
        pred_val = (prob_val >= 0.5).astype(int)
        val_roc = roc_auc_score(y_val, prob_val)
        val_pr = average_precision_score(y_val, prob_val)
        val_f1 = f1_score(y_val, pred_val)
        results.append((p, clf, val_roc, val_pr, val_f1))
        print(f'  {p}  valROC={val_roc:.4f}  valPR={val_pr:.4f}  valF1={val_f1:.4f}')

    rows = []
    for crit, idx in [('ROC-AUC', 2), ('PR-AUC', 3), ('F1', 4)]:
        best = max(results, key=lambda r: r[idx])
        best_params, best_clf = best[0], best[1]
        prob_test = best_clf.predict_proba(X_test)[:, 1]
        for cpt in [0.5, 0.8]:
            m = evaluate(y_test, prob_test, name, cpt)
            m['SelectionCriterion'] = crit
            m['WinningParams'] = str(best_params)
            rows.append(m)
    return rows


all_rows = []

all_rows += run_grid('Random Forest',
    ParameterGrid({'n_estimators': [300, 500], 'max_features': ['sqrt', 0.3],
                    'min_samples_leaf': [1, 3], 'max_depth': [None, 20]}),
    lambda p: RandomForestClassifier(**p, class_weight='balanced', n_jobs=-1, random_state=SEED))

all_rows += run_grid('SVM',
    ParameterGrid({'C': [1, 10, 50], 'gamma': ['scale', 0.01, 0.001]}),
    lambda p: SVC(**p, kernel='rbf', class_weight='balanced', probability=True, random_state=SEED))

sw = (y_train == 0).sum() / (y_train == 1).sum()
all_rows += run_grid('XGBoost',
    ParameterGrid({'n_estimators': [300, 500], 'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1], 'subsample': [0.8], 'colsample_bytree': [0.8]}),
    lambda p: XGBClassifier(**p, scale_pos_weight=sw, eval_metric='logloss',
                             n_jobs=-1, random_state=SEED, verbosity=0))

df = pd.DataFrame(all_rows)
cols = ['Model', 'SelectionCriterion', 'CPT'] + [c for c in METRIC_COLS if c not in ('Model', 'CPT')] + ['WinningParams']
df = df[cols]
df.to_csv('selection_criterion_comparison.csv', index=False)

print('\n\n=== SUMMARY: TEST-SET RESULTS BY SELECTION CRITERION (AID_1347037, Random Split) ===\n')
for cpt in [0.5, 0.8]:
    print(f'--- CPT={cpt} ---')
    sub = df[df.CPT == cpt][['Model', 'SelectionCriterion', 'Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'F1']]
    print(sub.to_string(index=False))
    print()
