"""Runs the TX2017-Structure pipeline on ONE split unit, for ONE stage.

Per unit:
  1. recover the unit's RAW training rows (clinical pool minus val minus the
     internal test split) and add NCV-888 for Stage 2
  2. select features on those raw rows: Random-Forest Gini importance, top 180
     bits -- the setting the original TX2017 search picked
  3. up-sample the training rows to 1:1 with imblearn's RandomOverSampler
     (TX2017's 'up' sampler), leaving val / internal test / Held-223 alone
  4. fit all five TX2017 classifiers and score val, the internal test split and
     the fixed Held-223 test set at cutoffs 0.50 and 0.80

Usage:
    python train_tx_unit.py <random_repeat|cv|scaffold> <1|2> <unit>
Env:
    TX_SMOKE=1      wiring test: small subsets, fewer trees, writes to results_smoke/
    TX_CLFS=NB,RF   restrict the classifier list

Outputs (results/<protocol>/):
    probs/<clf>_s<stage>_u<unit>_{test,itest}.npy
    <clf>_s<stage>_u<unit>_metrics.csv
    feat_s<stage>_u<unit>.npy           -- the selected bit indices
"""
import os
import sys
import time
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common_tx import (CLASSIFIERS, CPTS, DATA, FS_TOPN, PROTOCOLS, SAMPLING,
                       SEED, make_classifier, unit_dir)
from eval_utils import METRIC_COLS, evaluate

SMOKE = os.environ.get('TX_SMOKE') == '1'


def select_features(X, y, top_n, seed=SEED):
    """Random-Forest Gini importance, top N bits -- the winning TX2017 setting.

    Same estimator settings as ../DILI_tx2017/feature_selection.py: a plain
    RandomForestClassifier at the library defaults, fitted on the stage's raw
    (un-resampled) training rows only.
    """
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=20 if SMOKE else 100,
                                random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    return np.sort(order[:top_n])


def build_unit(protocol, stage, unit):
    """Training rows, plus val / internal test / Held-223.

    With TX_SAMPLING=up the training rows are the RAW (pre-up-sampling) ones.
    Every protocol partitions the clinical pool into train / val / internal
    test, so they are exactly the pool minus the other two; taking the
    complement rather than reading idx_train.npy keeps that path independent of
    whether the file on disk was written down- or up-sampled.

    With TX_SAMPLING=down the training rows are read straight from
    idx_train.npy, which ../DILI_split_protocols wrote already 1:1 downsampled
    (inactives drawn without replacement down to the active count, with
    RandomState(42 + unit)). Reusing that file rather than redrawing means
    TX2017 trains on the identical compounds as the graph and classical models
    in the downsampled tables, so the rows line up unit for unit.
    """
    Xc = np.load(f'{DATA}/X1024_clin.npy'); yc = np.load(f'{DATA}/y_clin.npy').astype(int)
    d = unit_dir(protocol, unit)
    va = np.load(f'{d}/idx_val.npy'); it = np.load(f'{d}/idx_itest.npy')
    held = np.union1d(va, it)
    if SAMPLING == 'down':
        tr = np.load(f'{d}/idx_train.npy')
        assert not np.intersect1d(tr, held).size, f'{protocol}/{unit}: train overlaps val/itest'
        n1 = int((yc[tr] == 1).sum())
        assert n1 * 2 == len(tr), (f'{protocol}/{unit}: idx_train.npy is not 1:1 '
                                   f'({n1} positive of {len(tr)}) -- is it the downsampled split?')
    else:
        tr = np.setdiff1d(np.arange(len(yc)), held)
        assert len(tr) + len(va) + len(it) == len(yc), f'{protocol}/{unit}: splits do not partition the pool'

    X_tr, y_tr = Xc[tr], yc[tr]
    if stage == 2:
        X_tr = np.vstack([X_tr, np.load(f'{DATA}/X1024_ncv.npy')])
        y_tr = np.concatenate([y_tr, np.load(f'{DATA}/y_ncv.npy').astype(int)])
    out = {'train': (X_tr, y_tr), 'val': (Xc[va], yc[va]), 'itest': (Xc[it], yc[it]),
           'test': (np.load(f'{DATA}/X1024_test.npy'), np.load(f'{DATA}/y_test.npy').astype(int))}
    if SMOKE:
        rng = np.random.RandomState(0)
        for k, m in (('train', 600), ('val', 200), ('itest', 200)):
            X, y = out[k]
            keep = rng.choice(len(y), min(len(y), m), replace=False)
            out[k] = (X[keep], y[keep])
    return out


def main(protocol, stage, unit):
    base = 'results' if SAMPLING == 'up' else 'results_down'
    out_root = os.path.join(HERE, 'results_smoke' if SMOKE else base, protocol)
    os.makedirs(f'{out_root}/probs', exist_ok=True)
    data = build_unit(protocol, stage, unit)
    X_tr, y_tr = data['train']

    feat = select_features(X_tr, y_tr, FS_TOPN)
    np.save(f'{out_root}/feat_s{stage}_u{unit}.npy', feat)

    if SAMPLING == 'up':
        from imblearn.over_sampling import RandomOverSampler
        Xs, ys = RandomOverSampler(random_state=SEED).fit_resample(X_tr[:, feat], y_tr)
        assert (ys == 1).sum() == (ys == 0).sum(), 'up-sampled training set is not 1:1'
        how = 'up-sampled'
    else:
        # The clinical rows arrived already 1:1 downsampled; Stage 2's NCV-888
        # rows are appended unresampled, exactly as in ../DILI_split_protocols,
        # so the stage-2 training set is 1:1 on its clinical part only.
        Xs, ys = X_tr[:, feat], y_tr
        how = 'downsampled'
    print(f'{protocol} stage {stage} unit {unit}: raw train {len(y_tr)} '
          f'(DILI+ {int(y_tr.sum())}) -> {how} {len(ys)} ({int((ys==1).sum())}/{int((ys==0).sum())}), '
          f'{len(feat)} of {X_tr.shape[1]} bits selected', flush=True)
    print(f'  val {len(data["val"][1])}, internal test {len(data["itest"][1])}, '
          f'Held-223 {len(data["test"][1])} (natural ratio, never resampled)', flush=True)

    names = os.environ.get('TX_CLFS')
    names = names.split(',') if names else CLASSIFIERS
    for name in names:
        t0 = time.time()
        clf = make_classifier(name)
        clf.fit(Xs, ys)
        probs = {s: clf.predict_proba(data[s][0][:, feat])[:, 1]
                 for s in ('val', 'itest', 'test')}
        for s in ('test', 'itest'):
            np.save(f'{out_root}/probs/{name}_s{stage}_u{unit}_{s}.npy',
                    probs[s].astype(np.float32))
        rows = [{**evaluate(data[s][1], probs[s], name, cpt), 'Protocol': protocol,
                 'Stage': stage, 'Unit': unit, 'Split': s}
                for cpt in CPTS for s in ('val', 'itest', 'test')]
        df = pd.DataFrame(rows)[['Protocol', 'Stage', 'Unit', 'Split'] + METRIC_COLS]
        df.to_csv(f'{out_root}/{name}_s{stage}_u{unit}_metrics.csv', index=False)
        t = df[(df.Split == 'test') & (df.CPT == 0.5)].iloc[0]
        print(f'  {name:8s} Held-223 ROC-AUC={t["ROC-AUC"]:.4f} CCR={t["CCR"]:.1f} '
              f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] not in PROTOCOLS or sys.argv[2] not in ('1', '2'):
        print(f'Usage: python {os.path.basename(__file__)} '
              f'<{"|".join(PROTOCOLS)}> <1|2> <unit>')
        sys.exit(1)
    p, st, u = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if not 0 <= u < PROTOCOLS[p]:
        print(f'unit must be 0..{PROTOCOLS[p] - 1} for {p}')
        sys.exit(1)
    main(p, st, u)
