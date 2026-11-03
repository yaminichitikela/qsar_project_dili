"""LaTeX Structure-only rows for the three 1:1-downsampled DILI tables.

Interval type per protocol is copied from
DILI_split_protocols/make_assay_style_tables.PROTO_SETS, so the rows match how
every other row in those tables is built:
    random_repeat -> SD across the 5 repeats
    cv, scaffold  -> Nadeau-Bengio corrected 95% half-width,
                     t(0.975, n-1) * sqrt((1/n + n_test/n_train) * var)

Usage: python make_rows.py [N ...]        default: 180 360
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
TX = os.path.join(REB, 'DILI_tx2017_splits')
DSP = os.path.join(REB, 'DILI_split_protocols')

N_TEST = 222
RULE = {'random_repeat': 'sd', 'cv': 'nb', 'scaffold': 'nb'}
UNITS = {'random_repeat': 5, 'cv': 10, 'scaffold': 5}
TABLE = {'random_repeat': 'tab:dili_random_assay_style (Part 1)',
         'cv': 'tab:dili_cv_assay_style (Part 3)',
         'scaffold': 'tab:dili_scaffold_assay_style (Part 5)'}
COLS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
PCT = {'Sensitivity', 'Specificity', 'CCR', 'Bal-Acc'}
N_NCV = len(np.load(f'{DSP}/data/y_ncv.npy'))


def n_train(protocol, stage):
    tr = [len(np.load(f'{TX}/splits/{protocol}/unit_{u}/idx_train.npy'))
          for u in range(UNITS[protocol])]
    return float(np.mean(tr)) + (N_NCV if stage == 2 else 0)


def interval(vals, kind, ntr):
    v = np.asarray(vals, float)
    n = len(v)
    sd = v.std(ddof=1)
    if kind == 'sd':
        return sd
    return stats.t.ppf(0.975, df=n - 1) * np.sqrt((1.0 / n + N_TEST / ntr) * sd ** 2)


def load(protocol, topn):
    """Per-unit rows for this protocol at this cutoff, whichever file holds them."""
    p = f'{HERE}/results/arm_n{topn}_{protocol}.csv'
    if os.path.exists(p):
        d = pd.read_csv(p)
    elif protocol == 'cv' and topn == 180:          # from the earlier arm-A run
        d = pd.read_csv(f'{HERE}/results/per_fold.csv')
        d = d[d.Arm == 'A_n180'].rename(columns={'Fold': 'Unit'})
    else:
        return None
    return d[d.CPT == 0.5]


def main():
    ns = [int(a) for a in sys.argv[1:]] or [180, 360]
    old = pd.read_csv(f'{TX}/results_down/tx_per_unit_all.csv')
    old = old[(old.Split == 'test') & (old.CPT == 0.5) & (old.Model == 'SVM')]
    for topn in ns:
        print('#' * 78)
        print(f'#  top-{topn} of 2048 bits  ({100 * topn / 2048:.1f}% of the fingerprint)')
        print('#' * 78)
        for protocol in ['random_repeat', 'cv', 'scaffold']:
            d = load(protocol, topn)
            if d is None:
                print(f'\n{TABLE[protocol]}: NOT RUN\n')
                continue
            print(f'\n% {TABLE[protocol]}   interval = {RULE[protocol].upper()}')
            for stage in (1, 2):
                s = d[d.Stage == stage]
                o = old[(old.Protocol == protocol) & (old.Stage == stage)]['ROC-AUC'].mean()
                cells = []
                for c in COLS:
                    m = s[c].mean()
                    if c in ('ROC-AUC', 'PR-AUC'):
                        half = interval(s[c].values, RULE[protocol], n_train(protocol, stage))
                        cells.append(f'${m:.3f} \\pm {half:.3f}$')
                    else:
                        cells.append(f'{m:.1f}' if c in PCT else f'{m:.3f}')
                print(f'%   Stage {stage}: published 1024/N=180 AUC {o:.3f} '
                      f'-> this row {s["ROC-AUC"].mean():.3f}')
                print('Structure only & ' + ' & '.join(cells) + r' \\')
        print()


if __name__ == '__main__':
    main()
