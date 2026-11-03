"""XGBoost -- grid search on val ROC-AUC, evaluate on val + test.
Requires prepare_split.py to have been run first (loads from ./data/).
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from eval_utils import evaluate, METRIC_COLS

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

sw = (y_train == 0).sum() / (y_train == 1).sum()
xgb_grid = ParameterGrid({
    'n_estimators': [300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
})

best_auc, best_clf, best_params = 0, None, {}
print(f'XGB: {len(xgb_grid)} configs...')
for p in xgb_grid:
    clf = XGBClassifier(**p, scale_pos_weight=sw, eval_metric='logloss',
                         n_jobs=-1, random_state=SEED, verbosity=0)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    if auc > best_auc:
        best_auc, best_clf, best_params = auc, clf, p
    print(f'  {p}  val AUC={auc:.4f}')

print(f'Best XGB val AUC={best_auc:.4f}  params={best_params}')

prob_val = best_clf.predict_proba(X_val)[:, 1]
prob_test = best_clf.predict_proba(X_test)[:, 1]

rows = []
for cpt in [0.5, 0.8]:
    rows.append({**evaluate(y_val, prob_val, 'XGBoost', cpt), 'Split': 'Val'})
    rows.append({**evaluate(y_test, prob_test, 'XGBoost', cpt), 'Split': 'Test'})

df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
df.to_csv(os.path.join(OUT_DIR, 'xgb_results.csv'), index=False)
np.save(os.path.join(OUT_DIR, 'xgb_test_probs.npy'), prob_test)
print(df)
print(f'Saved -> {OUT_DIR}/xgb_results.csv')
