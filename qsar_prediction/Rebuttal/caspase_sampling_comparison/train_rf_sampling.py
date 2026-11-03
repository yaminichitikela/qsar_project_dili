"""RF -- grid search on val ROC-AUC, evaluate on val + test. Self-contained
in this folder: reads ./data/<dataset>/<variant>/, writes
./results_sampling/<dataset>/<variant>/.

Usage: python train_rf_sampling.py <hepg2|cho_k1> <v1_ratio_1to5|v2_similarity|v3_diversity|v4_smote>
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

from common import CASPASE_DATASETS, DATA_ROOT
from eval_utils import evaluate, METRIC_COLS

VALID_VARIANTS = ['v1_ratio_1to5', 'v2_similarity', 'v3_diversity', 'v4_smote']

if len(sys.argv) != 3 or sys.argv[1] not in CASPASE_DATASETS or sys.argv[2] not in VALID_VARIANTS:
    print(f'Usage: python {sys.argv[0]} <{"|".join(CASPASE_DATASETS)}> <{"|".join(VALID_VARIANTS)}>')
    sys.exit(1)
DATASET_NAME, VARIANT = sys.argv[1], sys.argv[2]

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DATA_ROOT, DATASET_NAME, VARIANT)
OUT_DIR = os.path.join(HERE, 'results_sampling', DATASET_NAME, VARIANT)
os.makedirs(OUT_DIR, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
print(f'[{DATASET_NAME}/{VARIANT}] train={len(y_train):,} val={len(y_val):,} test={len(y_test):,}')

rf_grid = ParameterGrid({
    'n_estimators': [300, 500],
    'max_features': ['sqrt', 0.3],
    'min_samples_leaf': [1, 3],
    'max_depth': [None, 20],
})

best_auc, best_clf, best_params = 0, None, {}
print(f'RF: {len(rf_grid)} configs...')
for p in rf_grid:
    clf = RandomForestClassifier(**p, class_weight='balanced', n_jobs=-1, random_state=SEED)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    if auc > best_auc:
        best_auc, best_clf, best_params = auc, clf, p
    print(f'  {p}  val AUC={auc:.4f}')

print(f'Best RF val AUC={best_auc:.4f}  params={best_params}')

prob_val = best_clf.predict_proba(X_val)[:, 1]
prob_test = best_clf.predict_proba(X_test)[:, 1]

rows = []
for cpt in [0.5, 0.8]:
    rows.append({**evaluate(y_val, prob_val, 'Random Forest', cpt), 'Split': 'Val'})
    rows.append({**evaluate(y_test, prob_test, 'Random Forest', cpt), 'Split': 'Test'})

df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
df.to_csv(os.path.join(OUT_DIR, 'rf_results.csv'), index=False)
np.save(os.path.join(OUT_DIR, 'rf_test_probs.npy'), prob_test)
print(df)
print(f'Saved -> {OUT_DIR}/rf_results.csv')
