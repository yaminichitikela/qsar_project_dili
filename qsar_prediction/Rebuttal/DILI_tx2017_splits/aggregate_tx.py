"""Summarises the TX2017-Structure split-protocol runs and writes the tables.

Reads every per-unit metrics file, then for each protocol / stage / classifier
reports the mean and spread across units on the fixed Held-223 test set, with
the internal test split as a secondary result.

Spread follows the convention used by the other rebuttal tables:
    random_repeat   standard deviation across the 5 repeats
    cv, scaffold    Nadeau--Bengio corrected 95% half-width,
                    t(0.975, n-1) * sqrt((1/n + n_test/n_train) * var)
                    -- the same formula as compute_ci_significance.corrected_ci,
                    with n_test the 222 Held-223 compounds and n_train the mean
                    up-sampled training-set size.

Usage: python aggregate_tx.py [cutoff]        (default 0.5)
Outputs (results/):
    tx_per_unit_all.csv, tx_summary.csv,
    tables_tx2017_splits_<upsampled|downsampled>_cpt<NN>.tex
(Results go to results/ for TX_SAMPLING=up, results_down/ for TX_SAMPLING=down.)
"""
import glob
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common_tx import (CLASSIFIERS, DATA, FS_TOPN, PROTOCOLS, RESULTS, SAMPLING,
                       SAMPLING_TAG, STAGES, WINNER_CLF, unit_dir)

# Wording that differs between the two balancing schemes.
TRAIN_DESC = {'up': '1:1 up-sampled train', 'down': '1:1 downsampled train'}[SAMPLING]
BALANCE_DESC = {
    'up': 'training balanced with random over-sampling',
    'down': ('training balanced by 1:1 downsampling, reusing the same downsampled '
             'training rows as the graph and classical models'),
}[SAMPLING]
WINNER_NOTE = {
    'up': ('$^{\\ast}$SVM with the over-sampler is the combination the original '
           'TX2017 search selected.'),
    'down': ('$^{\\ast}$SVM is the classifier the original TX2017 search selected; '
             'that search paired it with over-sampling, so this table reports it '
             'under downsampling for comparability with the other downsampled tables.'),
}[SAMPLING]

COLS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
PCT = {'Sensitivity', 'Specificity', 'CCR', 'Bal-Acc'}
# Every metric carries its own interval, matching the DILI split-protocol
# tables these rows appear beside.
CI_COLS = set(COLS)
TEX_NAME = {'NB': 'Naive Bayes', 'SVM': 'SVM$^{\\ast}$', 'RF': 'Random Forest',
            'NNET': 'Neural network', 'XGBoost': 'XGBoost'}
PROTO = [
    ('random_repeat', 'repeated 80/10/10 split (mean $\\pm$ std across 5 repeats)', 'sd',
     'Intervals on every metric are the standard deviation across repeats; on the '
     'percentage columns they are in percentage points.', 'random'),
    ('cv', '10-fold cross-validation (mean across 10 folds)', 'nb', None, 'cv'),
    ('scaffold', 'scaffold split, 5 repeats (mean across 5 repeats)', 'nb', None, 'scaffold'),
]
NB_TEXT = ('Intervals on every metric are Nadeau--Bengio corrected 95\\% intervals '
           '(correction $1/n + n_{\\mathrm{test}}/n_{\\mathrm{train}}$, with $n_{\\mathrm{test}}$ '
           'the 222 Held-223 compounds and $n_{\\mathrm{train}}$ the mean training-set size); '
           'on the percentage columns they are in percentage points.')


def nd(c):
    """Displayed decimals: percentages to two, every other value to three."""
    return 2 if c in PCT else 3


def load():
    files = glob.glob(f'{RESULTS}/*/*_s[12]_u*_metrics.csv')
    if not files:
        print(f'no per-unit metrics under {RESULTS} -- run train_tx_unit.py first')
        sys.exit(1)
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def train_sizes(n_ncv):
    """Mean training-set size per protocol and stage, after balancing."""
    yc = np.load(f'{DATA}/y_clin.npy').astype(int)
    out = {}
    for proto, n in PROTOCOLS.items():
        for stage in STAGES:
            sizes = []
            for u in range(n):
                d = unit_dir(proto, u)
                held = np.union1d(np.load(f'{d}/idx_val.npy'), np.load(f'{d}/idx_itest.npy'))
                tr = np.setdiff1d(np.arange(len(yc)), held)
                if SAMPLING == 'down':
                    n_tr = len(np.load(f'{d}/idx_train.npy'))
                    if stage == 2:
                        n_tr += n_ncv           # NCV-888 is appended unresampled
                    sizes.append(n_tr)
                    continue
                y = yc[tr]
                if stage == 2:
                    y = np.concatenate([y, np.load(f'{DATA}/y_ncv.npy').astype(int)])
                sizes.append(2 * max(int((y == 1).sum()), int((y == 0).sum())))
            out[(proto, stage)] = float(np.mean(sizes))
    return out


def interval(vals, kind, ntr, nte):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    sd = vals.std(ddof=1) if n > 1 else 0.0
    if kind == 'sd' or n < 2:
        return sd
    se = np.sqrt((1.0 / n + nte / ntr) * sd ** 2)
    return stats.t.ppf(0.975, df=n - 1) * se


def table(per, proto, desc, kind, int_text, tag, cpt, ntr, nte):
    L = ['\\begin{table*}[t]', '\\centering',
         f'\\caption{{TX2017-Structure performance on the FDA-approved DILI held-out test set '
         f'(Held-223; 222 compounds, 155 hepatotoxic/67 non-hepatotoxic) -- {desc}, '
         f'{TRAIN_DESC}; Stage 1 (Proxy-only) and Stage 2 (Proxy+DILI); '
         f'classification cutoff {cpt:.1f}. Structure-only ECFP4 (1024 bits), '
         f'Random-Forest Gini feature selection (top {FS_TOPN} bits, re-selected inside each '
         f'unit from its training rows alone), {BALANCE_DESC}; '
         f'validation and Held-223 keep their natural class ratio. {int_text or NB_TEXT} '
         f'Percentages are given to two decimals and all other values to three. '
         f'Best value per column within each stage in bold, second best underlined. '
         f'{WINNER_NOTE}}}',
         f"\\label{{tab:tx2017_{tag}_{'up' if SAMPLING == 'up' else 'down'}}}",
         '\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{lcccccccc}', '\\toprule',
         '\\textbf{Model} & \\textbf{Sensitivity} & \\textbf{Specificity} & \\textbf{CCR} & '
         '\\textbf{ROC-AUC} & \\textbf{PR-AUC} & \\textbf{MCC} & \\textbf{Bal-Acc} & \\textbf{F1} \\\\',
         ' & (\\%) & (\\%) & (\\%) & & & & (\\%) & \\\\']
    for stage, sname in ((1, 'Stage 1 (Proxy-only)'), (2, 'Stage 2 (Proxy+DILI)')):
        sub = per[(per.Protocol == proto) & (per.Stage == stage)]
        cells = {}
        for clf in CLASSIFIERS:
            g = sub[sub.Model == clf]
            if g.empty:
                continue
            cells[clf] = {c: round(float(g[c].mean()), nd(c)) for c in COLS}
            for c in CI_COLS:
                cells[clf][c + '_ci'] = interval(g[c].values, kind, ntr[(proto, stage)], nte)
        if not cells:
            continue
        # Ranked on the DISPLAYED value; ties share a rank, so every cell equal
        # to the best is bold and every cell equal to the next distinct value
        # is underlined.
        ranked = {c: sorted({v[c] for v in cells.values()}, reverse=True) for c in COLS}
        best = {c: r[0] for c, r in ranked.items()}
        second = {c: (r[1] if len(r) > 1 else None) for c, r in ranked.items()}
        L += ['\\midrule', f'\\multicolumn{{9}}{{c}}{{\\textbf{{{sname}}}}} \\\\', '\\midrule']
        for clf in CLASSIFIERS:
            if clf not in cells:
                continue
            row = []
            for c in COLS:
                v = cells[clf][c]
                s = (f'${v:.{nd(c)}f} \\pm {cells[clf][c + "_ci"]:.{nd(c)}f}$'
                     if c in CI_COLS else f'{v:.{nd(c)}f}')
                if v == best[c]:
                    s = f'\\textbf{{\\boldmath{s}}}' if c in CI_COLS else f'\\textbf{{{s}}}'
                elif v == second[c]:
                    s = f'\\underline{{{s}}}'
                row.append(s)
            L.append(f'{TEX_NAME[clf]} & ' + ' & '.join(row) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}%', '}', '\\end{table*}', '']
    return '\n'.join(L)


def main():
    cpt = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    allrows = load()
    allrows.to_csv(f'{RESULTS}/tx_per_unit_all.csv', index=False)
    per = allrows[(allrows.Split == 'test') & ((allrows.CPT - cpt).abs() < 1e-9)]

    rows = []
    for (proto, stage, clf, c), g in allrows[allrows.Split.isin(['test', 'itest'])].groupby(
            ['Protocol', 'Stage', 'Model', 'CPT']):
        for split, gg in g.groupby('Split'):
            r = {'Protocol': proto, 'Split': split, 'Stage': stage, 'Model': clf,
                 'CPT': c, 'n_units': gg.Unit.nunique()}
            for m in COLS:
                r[f'{m}_mean'] = round(float(gg[m].mean()), 4)
                r[f'{m}_sd'] = round(float(gg[m].std(ddof=1)) if len(gg) > 1 else 0.0, 4)
            rows.append(r)
    summ = pd.DataFrame(rows)
    summ.to_csv(f'{RESULTS}/tx_summary.csv', index=False)

    nte = len(np.load(f'{DATA}/y_test.npy'))
    ntr = train_sizes(len(np.load(f'{DATA}/y_ncv.npy')))
    out = [f'% TX2017-Structure under the three split protocols, {TRAIN_DESC}, '
           f'cutoff {cpt:.2f}.',
           '% Generated by DILI_tx2017_splits/aggregate_tx.py; needs booktabs, graphicx, amsmath.',
           '']
    out += [table(per, p, d, k, t, tag, cpt, ntr, nte) for p, d, k, t, tag in PROTO]
    path = f'{RESULTS}/tables_tx2017_splits_{SAMPLING_TAG}_cpt{int(round(cpt * 100))}.tex'
    with open(path, 'w') as fh:
        fh.write('\n'.join(out))

    pd.set_option('display.width', 200)
    print(f'{len(per)} Held-223 rows at cutoff {cpt}; units per protocol: '
          + ', '.join(f'{p}={per[per.Protocol == p].Unit.nunique()}' for p in PROTOCOLS))
    w = per[per.Model == WINNER_CLF]
    print(f'\n{WINNER_CLF} (the TX2017 winning classifier), Held-223 ROC-AUC:')
    print(w.groupby(['Protocol', 'Stage'])['ROC-AUC'].agg(['mean', 'std', 'count']).round(4).to_string())
    print('\nAll classifiers, Held-223 ROC-AUC mean:')
    print(per.pivot_table(index='Model', columns=['Protocol', 'Stage'],
                          values='ROC-AUC', aggfunc='mean').round(4).to_string())
    print(f'\n-> {RESULTS}/tx_summary.csv')
    print(f'-> {path}')


if __name__ == '__main__':
    main()
