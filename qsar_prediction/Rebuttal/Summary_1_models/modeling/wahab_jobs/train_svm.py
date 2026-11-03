"""SVM -- grid search on val ROC-AUC, evaluate on val + test.
Requires prepare_split.py to have been run first (loads from ./data/).
"""
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
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

svm_grid = ParameterGrid({
    'C': [1, 10, 50],
    'gamma': ['scale', 0.01, 0.001],
})

best_auc, best_clf, best_params = 0, None, {}
print(f'SVM: {len(svm_grid)} configs...')
for p in svm_grid:
    clf = SVC(**p, kernel='rbf', class_weight='balanced', probability=True, random_state=SEED)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    if auc > best_auc:
        best_auc, best_clf, best_params = auc, clf, p
    print(f'  {p}  val AUC={auc:.4f}')

print(f'Best SVM val AUC={best_auc:.4f}  params={best_params}')

prob_val = best_clf.predict_proba(X_val)[:, 1]
prob_test = best_clf.predict_proba(X_test)[:, 1]

rows = []
for cpt in [0.5, 0.8]:
    rows.append({**evaluate(y_val, prob_val, 'SVM', cpt), 'Split': 'Val'})
    rows.append({**evaluate(y_test, prob_test, 'SVM', cpt), 'Split': 'Test'})

df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
df.to_csv(os.path.join(OUT_DIR, 'svm_results.csv'), index=False)
np.save(os.path.join(OUT_DIR, 'svm_test_probs.npy'), prob_test)
print(df)
print(f'Saved -> {OUT_DIR}/svm_results.csv')
