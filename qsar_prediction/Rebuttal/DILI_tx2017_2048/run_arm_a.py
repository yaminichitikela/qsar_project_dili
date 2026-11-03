"""Arm A: Xu et al. "Structure only" method on the 2048-bit ECFP4, for all
three 1:1-downsampled DILI split protocols.

Xu's method is kept intact -- RF-Gini bit ranking, a top-N cutoff, a stock RBF
SVM, and the same 1:1 downsampled training rows the other models use. The only
change from the published row is the fingerprint: 2048 bits instead of the 1024
Xu et al. specify, so the representation matches every other model in the table.

THE CUTOFF
----------
Xu et al. select the top 180 of 1024 bits, i.e. 17.6% of the fingerprint. Held
at 180 on a 2048-bit fingerprint that proportion halves to 8.8%, which makes the
selection markedly more aggressive than the original rather than equivalent to
it. --n 360 preserves the original proportion at the new bit length; --n 180
reproduces the literal cutoff. Both are reported.

The 1024- and 2048-bit arrays are row-aligned (identical InChIKeys and labels),
so the existing split indices apply unchanged.

Usage: python run_arm_a.py [--n N] [protocol ...]
       default: --n 180, protocols random_repeat scaffold
Output: results/arm_n<N>_<protocol>.csv
"""
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
from eval_utils import evaluate  # noqa: E402

SEED = 42
CPTS = (0.5, 0.8)
UNITS = {'random_repeat': 5, 'cv': 10, 'scaffold': 5}


def build(protocol, stage, unit):
    Xc = np.load(f'{DSP}/data/X_clin.npy')
    yc = np.load(f'{DSP}/data/y_clin.npy').astype(int)
    d = f'{TX}/splits/{protocol}/unit_{unit}'
    tr = np.load(f'{d}/idx_train.npy')
    va, it = np.load(f'{d}/idx_val.npy'), np.load(f'{d}/idx_itest.npy')
    assert not np.intersect1d(tr, np.union1d(va, it)).size, f'{protocol}/{unit}: train overlaps'
    n1 = int((yc[tr] == 1).sum())
    assert n1 * 2 == len(tr), f'{protocol}/{unit}: idx_train not 1:1 ({n1} of {len(tr)})'
    X_tr, y_tr = Xc[tr], yc[tr]
    if stage == 2:
        X_tr = np.vstack([X_tr, np.load(f'{DSP}/data/X_ncv.npy')])
        y_tr = np.concatenate([y_tr, np.load(f'{DSP}/data/y_ncv.npy').astype(int)])
    return X_tr, y_tr, np.load(f'{DSP}/data/X_test.npy'), np.load(f'{DSP}/data/y_test.npy').astype(int)


def main():
    args = sys.argv[1:]
    topn = 180
    if '--n' in args:
        i = args.index('--n'); topn = int(args[i + 1]); del args[i:i + 2]
    protos = args or ['random_repeat', 'scaffold']
    print(f'top-N cutoff = {topn} of 2048 bits ({100 * topn / 2048:.1f}%)\n')
    os.makedirs(f'{HERE}/results', exist_ok=True)
    for protocol in protos:
        rows = []
        for stage in (1, 2):
            for unit in range(UNITS[protocol]):
                t0 = time.time()
                X_tr, y_tr, X_te, y_te = build(protocol, stage, unit)
                rf = RandomForestClassifier(n_estimators=100, random_state=SEED,
                                            n_jobs=-1).fit(X_tr, y_tr)
                feat = np.sort(np.argsort(rf.feature_importances_)[::-1][:topn])
                p = SVC(kernel='rbf', probability=True, random_state=SEED) \
                    .fit(X_tr[:, feat], y_tr).predict_proba(X_te[:, feat])[:, 1]
                for cpt in CPTS:
                    rows.append({'Protocol': protocol, 'N': topn, 'Stage': stage,
                                 'Unit': unit, **evaluate(y_te, p, 'Structure only', cpt)})
                print(f'{protocol} s{stage} u{unit}: ROC-AUC={rows[-1]["ROC-AUC"]:.4f} '
                      f'({time.time() - t0:.0f}s)', flush=True)
        out = f'{HERE}/results/arm_n{topn}_{protocol}.csv'
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f'wrote {out}\n')


if __name__ == '__main__':
    main()
