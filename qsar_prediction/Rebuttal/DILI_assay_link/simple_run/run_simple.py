"""Simple route: can the assay models predict clinical DILI?

THE IDEA
--------
Train a model on an in vitro assay (MMP or caspase-3/7), then use it to score
the 222 compounds of the DILI held-out test set, and see how well that score
predicts the clinical DILI label.

WHY THIS IS SO MUCH SIMPLER THAN ../build_invitro_features.py
-------------------------------------------------------------
That pipeline needs clean assay predictions for all 13,241 DILI compounds,
because the DILI models TRAIN on 12,145 of them. Excluding that many from
assay training would leave MMP with 1,424 compounds, so it uses out-of-bag
estimation over 21 repeated splits instead.

Here we only score the 222 TEST compounds. The compounds to exclude are
therefore just the test compounds that happen to sit in the assay data:
62 of 5,937 for MMP, 79 of 7,282 for caspase. Dropping ~1% of the assay data
costs nothing, so a single straightforward train/predict is enough and no
out-of-bag machinery is needed.

WHAT IT DOES
------------
  1. load the assay compounds (fingerprints + active/inactive labels)
  2. drop any that also appear in the DILI test set
  3. train on what is left
  4. predict the 222 DILI test compounds
  5. score those predictions against the clinical DILI label

Models are the classical families, which need no GPU. Hyperparameters come
from ../results/<assay>/<FAMILY>_winning_config.json when present, otherwise
sensible defaults.

Usage:
    python run_simple.py                  # RF, SVM, XGB on both assays
    python run_simple.py --models RF      # just one
Outputs (results/):
    predictions.csv   one row per test compound per model
    scores.csv        one row per model: ROC-AUC, PR-AUC, sensitivity, ...
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             matthews_corrcoef, roc_auc_score)
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
LINK = os.path.dirname(HERE)          # ../DILI_assay_link
OUT = os.path.join(HERE, 'results')
SEED = 42
ASSAYS = ['mmp', 'caspase']
CUTOFF = 0.5


# ── 1. load ──────────────────────────────────────────────────────────────────
def load_assay(assay):
    """Fingerprints, labels and InChIKeys for one assay."""
    X = np.load(f'{LINK}/data_{assay}/X_all.npy')
    y = np.load(f'{LINK}/data_{assay}/y_all.npy').astype(int)
    keys = [l.strip() for l in open(f'{LINK}/data_{assay}/ikeys_all.txt') if l.strip()]
    assert len(keys) == len(y) == X.shape[0]
    return X, y, np.array(keys)


def load_dili_test():
    """Fingerprints and DILI labels for the 222 held-out test compounds."""
    X = np.load(f'{LINK}/data_dili_features/X_dili.npy')
    keys = np.array([l.strip() for l in open(f'{LINK}/data_dili_features/ikeys_dili.txt')
                     if l.strip()])
    lab = pd.read_csv(f'{LINK}/data_dili_features/dili_labels.csv')
    lab = lab[lab.dili_set == 'Held-223']
    pos = {k: i for i, k in enumerate(keys)}
    rows = [pos[k] for k in lab.InChIKey if k in pos]
    keep = [k in pos for k in lab.InChIKey]
    return X[rows], lab.TOXICITY.values[keep].astype(int), lab.InChIKey.values[keep]


# ── 2. models ────────────────────────────────────────────────────────────────
def make_model(name, y_train):
    """Winning config if one was saved, otherwise a sensible default."""
    path = f'{LINK}/results/{{}}/{name}_winning_config.json'
    params = {}
    for assay in ASSAYS:                       # configs are per assay; take any
        p = path.format(assay)
        if os.path.exists(p):
            params = json.load(open(p))
            break
    if name == 'RF':
        return RandomForestClassifier(**params, class_weight='balanced',
                                      n_jobs=-1, random_state=SEED)
    if name == 'SVM':
        return SVC(**params, kernel='rbf', class_weight='balanced',
                   probability=True, random_state=SEED)
    if name == 'XGB':
        pos = max(int((y_train == 1).sum()), 1)
        return XGBClassifier(**params, scale_pos_weight=(y_train == 0).sum() / pos,
                             eval_metric='logloss', n_jobs=-1,
                             random_state=SEED, verbosity=0)
    raise ValueError(name)


# ── 3. scoring ───────────────────────────────────────────────────────────────
def score(y_true, prob, cutoff=CUTOFF):
    pred = (prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    sens = 100 * tp / (tp + fn) if tp + fn else 0.0
    spec = 100 * tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {'Sensitivity': round(sens, 1), 'Specificity': round(spec, 1),
            'CCR': round((sens + spec) / 2, 1),
            'ROC-AUC': round(roc_auc_score(y_true, prob), 3),
            'PR-AUC': round(average_precision_score(y_true, prob), 3),
            'MCC': round(matthews_corrcoef(y_true, pred), 3),
            'F1': round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0}


# ── 4. main ──────────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    models = (args[args.index('--models') + 1].split(',')
              if '--models' in args else ['RF', 'SVM', 'XGB'])
    os.makedirs(OUT, exist_ok=True)

    X_test, y_test, test_keys = load_dili_test()
    print(f'DILI test set: {len(y_test)} compounds, {y_test.sum()} hepatotoxic '
          f'({y_test.mean():.1%})\n')

    scores, preds = [], pd.DataFrame({'InChIKey': test_keys, 'DILI': y_test})
    for assay in ASSAYS:
        X, y, keys = load_assay(assay)

        # the only leakage control needed: drop test compounds from assay training
        overlap = np.isin(keys, test_keys)
        X_tr, y_tr = X[~overlap], y[~overlap]
        print(f'{assay.upper()}: {len(y)} compounds, {overlap.sum()} also in the DILI '
              f'test set -> train on {len(y_tr)} ({len(y_tr) / len(y):.1%})')
        assert not np.isin(keys[~overlap], test_keys).any(), 'overlap not removed'

        probs = {}
        for name in models:
            clf = make_model(name, y_tr).fit(X_tr, y_tr)
            p = clf.predict_proba(X_test)[:, 1]
            probs[name] = p
            preds[f'{assay}_{name}'] = p
            scores.append({'Assay': assay, 'Model': name, **score(y_test, p)})
            print(f'   {name:4s} ROC-AUC={scores[-1]["ROC-AUC"]:.3f}  '
                  f'PR-AUC={scores[-1]["PR-AUC"]:.3f}')

        if len(probs) > 1:                      # consensus = mean of the families
            p = np.mean(list(probs.values()), axis=0)
            preds[f'{assay}_Consensus'] = p
            scores.append({'Assay': assay, 'Model': 'Consensus', **score(y_test, p)})
            print(f'   {"Consensus":9s} ROC-AUC={scores[-1]["ROC-AUC"]:.3f}  '
                  f'PR-AUC={scores[-1]["PR-AUC"]:.3f}')
        print()

    df = pd.DataFrame(scores)
    df.to_csv(f'{OUT}/scores.csv', index=False)
    preds.to_csv(f'{OUT}/predictions.csv', index=False)
    print(df.to_string(index=False))
    print(f'\nbaseline: always-positive gives PR-AUC {y_test.mean():.3f}, '
          f'ROC-AUC 0.500')
    print(f'wrote {OUT}/scores.csv and predictions.csv')


if __name__ == '__main__':
    main()
