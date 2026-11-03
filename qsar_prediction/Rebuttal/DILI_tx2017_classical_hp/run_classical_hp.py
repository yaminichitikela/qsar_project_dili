"""Structure-only features trained with THIS PAPER'S classical hyperparameters.

Why
---
In the SI tables the replicated Xu et al. "Structure only" row sits below
Random Forest and XGBoost, which invites the question of whether the structural
feature set is genuinely weaker or whether the gap is just the pipeline around
it. The two rows differ in three confounded ways at once:

    fingerprint      1024 bits            vs  2048 bits
    feature set      top-180 RF-Gini      vs  all bits
    model            TX2017 stock SVM     vs  per-family Optuna + balanced
                     + random oversampling    class weights

This script holds the splits fixed and varies only the last two, so the gap can
be attributed. Four arms, two of them already on disk:

    A  1024 -> top-180   TX2017 stock SVM        existing: DILI_tx2017_splits/results_down
    B  1024 -> top-180   classical hyperparams   COMPUTED HERE
    C  1024 -> all bits  classical hyperparams   COMPUTED HERE
    D  2048 -> all bits  classical hyperparams   existing: DILI_split_protocols/results

A vs B isolates the model/balancing; B vs C isolates the 180-bit selection;
C vs D isolates the fingerprint length.

Everything is read from the existing runs: the same split indices, the same
1:1 downsampled idx_train.npy, the same Held-223 arrays, and the same winning
configs in DILI_split_protocols/configs. Nothing here overwrites any existing
result; output goes to this folder's results/.

Usage:
    python run_classical_hp.py            # all protocols, both stages
    python run_classical_hp.py --quick    # cv unit 0 only, for timing
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
TX = os.path.join(REB, 'DILI_tx2017_splits')
DSP = os.path.join(REB, 'DILI_split_protocols')
sys.path.insert(0, TX)
from eval_utils import evaluate  # noqa: E402  (same metric code as every other run)

SEED = 42
FS_TOPN = 180
PROTOCOLS = {'random_repeat': 5, 'cv': 10, 'scaffold': 5}
FAMILIES = ['RF', 'SVM', 'XGB']
CPTS = (0.5, 0.8)
OUT = os.path.join(HERE, 'results')


def cfg(family, stage):
    with open(os.path.join(DSP, 'configs', f'{family}_s{stage}_winning_config.json')) as fh:
        return json.load(fh)


def make_classical(family, params, y_train):
    """Byte-for-byte the construction in DILI_split_protocols/trainer.py."""
    if family == 'RF':
        return RandomForestClassifier(**params, class_weight='balanced',
                                      n_jobs=-1, random_state=SEED)
    if family == 'SVM':
        return SVC(**params, kernel='rbf', class_weight='balanced',
                   probability=True, random_state=SEED)
    if family == 'XGB':
        from xgboost import XGBClassifier
        pos = max(int((y_train == 1).sum()), 1)
        sw = float((y_train == 0).sum()) / pos
        return XGBClassifier(**params, scale_pos_weight=sw, eval_metric='logloss',
                             n_jobs=-1, random_state=SEED, verbosity=0)
    raise ValueError(family)


def select_features(X, y, top_n):
    """RF Gini importance, top N bits -- identical to DILI_tx2017_splits."""
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    return np.sort(np.argsort(rf.feature_importances_)[::-1][:top_n])


def build_unit(protocol, stage, unit):
    """1024-bit training/Held-223 arrays on the existing 1:1 downsampled rows."""
    data = os.path.join(TX, 'data_tx')
    Xc = np.load(f'{data}/X1024_clin.npy')
    yc = np.load(f'{data}/y_clin.npy').astype(int)
    d = os.path.join(TX, 'splits', protocol, f'unit_{unit}')
    tr = np.load(f'{d}/idx_train.npy')
    va, it = np.load(f'{d}/idx_val.npy'), np.load(f'{d}/idx_itest.npy')
    assert not np.intersect1d(tr, np.union1d(va, it)).size, 'train overlaps val/itest'
    n1 = int((yc[tr] == 1).sum())
    assert n1 * 2 == len(tr), f'idx_train.npy is not 1:1 ({n1} of {len(tr)})'

    X_tr, y_tr = Xc[tr], yc[tr]
    if stage == 2:
        X_tr = np.vstack([X_tr, np.load(f'{data}/X1024_ncv.npy')])
        y_tr = np.concatenate([y_tr, np.load(f'{data}/y_ncv.npy').astype(int)])
    return (X_tr, y_tr,
            np.load(f'{data}/X1024_test.npy'),
            np.load(f'{data}/y_test.npy').astype(int))


def main():
    quick = '--quick' in sys.argv[1:]
    todo = [('cv', [0])] if quick else [(p, list(range(n))) for p, n in PROTOCOLS.items()]
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for protocol, units in todo:
        for stage in (1, 2):
            for unit in units:
                X_tr, y_tr, X_te, y_te = build_unit(protocol, stage, unit)
                feat = select_features(X_tr, y_tr, FS_TOPN)
                arms = {'B_top180': feat, 'C_allbits': np.arange(X_tr.shape[1])}
                for arm, cols in arms.items():
                    for fam in FAMILIES:
                        t0 = time.time()
                        clf = make_classical(fam, cfg(fam, stage), y_tr)
                        clf.fit(X_tr[:, cols], y_tr)
                        p = clf.predict_proba(X_te[:, cols])[:, 1]
                        for cpt in CPTS:
                            rows.append({'Arm': arm, 'Protocol': protocol, 'Stage': stage,
                                         'Unit': unit, 'Split': 'test',
                                         **evaluate(y_te, p, fam, cpt)})
                        print(f'{arm} {protocol} s{stage} u{unit} {fam:4s} '
                              f'{len(cols):4d} bits  AUC={rows[-1]["ROC-AUC"]:.4f} '
                              f'({time.time() - t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'per_unit_quick.csv' if quick else 'per_unit_all.csv')
    df.to_csv(path, index=False)
    print(f'\nwrote {path}  ({len(df)} rows)')


if __name__ == '__main__':
    main()
