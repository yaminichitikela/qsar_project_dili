"""Runs ONE feature-selection setting's 20 model fits (4 samplers x 5
classifiers) for ONE stage, so all settings x both stages can run as separate
parallel jobs instead of one long sequential script.

Two-stage protocol: Stage-1 ("Proxy-only") trains on clin_train alone;
Stage-2 ("Proxy+DILI") trains on clin_train + NCV-888 combined. Both
validate on the same clin_val. feat_idx/settings come from that stage's own
feature_selection.json (results/stage<N>/feature_selection.json) -- feature
selection was run independently per stage, not shared.

Speed fix vs. the original design: SVM is fit with probability=False during
this search phase and ranked via decision_function() instead of
predict_proba() -- ROC-AUC is invariant to monotonic transforms of the
score, so this gives the identical val-AUC ranking as the slow, fully
probability-calibrated SVM would, without paying for SVC(probability=True)'s
expensive internal 5-fold Platt-scaling calibration on every fit. The
winning combination gets re-fit with full probability calibration exactly
once, in aggregate_results.py, only if it turns out to be SVM.

Results are written immediately after this setting finishes (checkpointed),
so killing/restarting never loses more than one setting's worth of work.

Usage: python train_one_setting.py <1|2> <setting_idx>
Requires prepare_features_clean.py and feature_selection.py <stage> to have
been run first.
"""
import os
import sys
import time
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from settings_list import load_settings

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

XGB_PARAMS = dict(
    objective='binary:logistic', eval_metric='logloss',
    max_depth=3, learning_rate=0.01, colsample_bytree=0.5,
    gamma=1, min_child_weight=1, n_estimators=100,
    n_jobs=-1, random_state=SEED, verbosity=0,
)

CLASSIFIERS = {
    'NB': lambda: BernoulliNB(alpha=1.0),  # alpha=1.0 = Laplace smoothing, matching the paper
    'SVM': lambda: SVC(kernel='rbf', probability=False, random_state=SEED),
    'RF': lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
    'NNET': lambda: MLPClassifier(random_state=SEED, max_iter=500),
    'XGBoost': lambda: XGBClassifier(**XGB_PARAMS),
}

SAMPLERS = {
    'original': None,
    'down': lambda: RandomUnderSampler(random_state=SEED),
    'up': lambda: RandomOverSampler(random_state=SEED),
    'rose': lambda: SMOTE(random_state=SEED),
}


def score(clf_name, clf, X):
    if clf_name == 'SVM':
        return clf.decision_function(X)
    return clf.predict_proba(X)[:, 1]


def load_stage_train(stage):
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    if stage == 2:
        X_ncv = np.load(os.path.join(DATA_DIR, 'X_ncv.npy'))
        y_ncv = np.load(os.path.join(DATA_DIR, 'y_ncv.npy'))
        X_train = np.vstack([X_train, X_ncv])
        y_train = np.concatenate([y_train, y_ncv])
    return X_train, y_train


def main(stage, setting_idx):
    settings = load_settings(stage)
    if not (0 <= setting_idx < len(settings)):
        print(f'setting_idx must be in [0, {len(settings)-1}]')
        sys.exit(1)
    method, cutoff, feat_idx = settings[setting_idx]
    print(f'Stage {stage}  Setting {setting_idx}: method={method} cutoff={cutoff} n_features={len(feat_idx)}')

    X_train, y_train = load_stage_train(stage)
    X_train = X_train[:, feat_idx]
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))[:, feat_idx]
    y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))

    rows = []
    t0 = time.time()
    for sampler_name, sampler_fn in SAMPLERS.items():
        if sampler_fn is None:
            Xs, ys = X_train, y_train
        else:
            try:
                Xs, ys = sampler_fn().fit_resample(X_train, y_train)
            except Exception as e:
                print(f'  {sampler_name}: sampler FAILED ({e}), skipping')
                continue

        for clf_name, clf_fn in CLASSIFIERS.items():
            tt = time.time()
            try:
                clf = clf_fn()
                clf.fit(Xs, ys)
                val_score = score(clf_name, clf, X_val)
                val_auc = roc_auc_score(y_val, val_score)
            except Exception as e:
                print(f'  {sampler_name}/{clf_name}: FAILED ({e})')
                continue
            rows.append({'Method': method, 'Cutoff': cutoff, 'Sampler': sampler_name,
                         'Classifier': clf_name, 'NumFeatures': len(feat_idx), 'ValAUC': val_auc})
            print(f'  {sampler_name}/{clf_name}: n_train={len(ys)} val_auc={val_auc:.4f} ({time.time()-tt:.1f}s)')

    settings_dir = os.path.join(RESULTS_DIR, f'stage{stage}', 'settings')
    os.makedirs(settings_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = os.path.join(settings_dir, f'setting_{setting_idx}.csv')
    df.to_csv(out_path, index=False)
    print(f'\nStage {stage} Setting {setting_idx} done in {time.time()-t0:.0f}s -> {out_path}')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('1', '2') or not sys.argv[2].isdigit():
        print(f'Usage: python {sys.argv[0]} <1|2> <setting_idx>')
        sys.exit(1)
    main(int(sys.argv[1]), int(sys.argv[2]))
