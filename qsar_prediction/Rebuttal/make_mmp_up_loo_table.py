"""Rebuilds Table tab:mmp_up_loo with 95% intervals on ALL eight metrics.

The published table carries intervals on ROC-AUC and PR-AUC only. The editor
asked for intervals on "all performance values presented", so every column now
gets one, computed from the same ten test-fold scores the means come from.

Interval: Nadeau--Bengio corrected 95% half-width,
    t(0.975, n-1) * sqrt((1/n + n_test/n_train) * var)
with n_test = N/10 (one held-out fold) and n_train = 0.8*N, the number of
DISTINCT training compounds. Up-sampling duplicates actives, so the row count
is larger, but the correction is about distinct compounds -- the same basis
compute_significance_upsampled.py argues for. That this reproduces the already
published ROC-AUC and PR-AUC intervals exactly is asserted below.

MEANS ARE TAKEN FROM THE PUBLISHED TABLE, AND THAT IS DELIBERATE
----------------------------------------------------------------
The per-fold CSVs store the percentage columns rounded to one decimal, while
ROC-AUC / PR-AUC / MCC / F1 are stored to four. Re-averaging the rounded
percentages reproduces the published value only to about +/-0.02, which is
exactly the error expected from averaging ten values quantised at 0.1
(sd = 0.1/sqrt(12)/sqrt(10) = 0.009). The published table was computed from the
unrounded fold values, so IT is the accurate one and a recomputation from disk
is slightly worse. The confirmation is that the four-decimal columns reproduce
exactly -- all 28 published ROC-AUC and PR-AUC intervals come back identical.

So the means are parsed from 1__editing.tex and reprinted untouched; only the
interval is computed here. The half-widths are unaffected by the quantisation:
rounding to 0.1 adds 0.1^2/12 = 0.0008 to a variance of order 20.

Usage: python make_mmp_up_loo_table.py
Output: results/table_mmp_up_loo_allci.tex
"""
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'Summary_1_models', 'upsampling_protocols', 'results_loo')
N_TOTAL = 5937
CPTS = (0.5, 0.8)
COLS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
PCT = {'Sensitivity', 'Specificity', 'CCR', 'Bal-Acc'}
FREE = {'ROC-AUC', 'PR-AUC'}          # threshold-free: one interval spans both cutoffs

# (group heading, [(file key, Model value in that file, printed label), ...])
GROUPS = [
    ('Classical Machine Learning', [
        ('rf', 'Random Forest', 'Random Forest'),
        ('svm', 'SVM', 'SVM'),
        ('xgb', 'XGBoost', 'XGBoost')]),
    ('Graph Neural Networks', [
        ('gcn', 'GCN', 'GCN'), ('gat', 'GAT', 'GAT'),
        ('gin', 'GIN', 'GIN'), ('sage', 'SAGE', 'SAGE')]),
    ('Multi-View Graph Neural Network', [
        ('mvgnn', 'MV-GNN-cross', 'MV-GNN$^{cross}$')]),
    ('Graph Transformers', [
        ('graphormer', 'Graphormer', 'Graphormer'),
        ('graphgps', 'GraphGPS', 'GraphGPS')]),
    ('Consensus Ensembles', [
        ('consensus', 'Consensus-Classical', 'Consensus-Classical'),
        ('consensus', 'Consensus-GNN', 'Consensus-GNN'),
        ('consensus', 'Consensus-GT', 'Consensus-GT'),
        ('consensus', 'Consensus', 'Consensus (Full)')]),
]
# the intervals already in the manuscript, to prove the convention is unchanged
PUBLISHED = {
    'Random Forest': ('0.897 \\pm 0.026', '0.678 \\pm 0.051'),
    'SVM': ('0.903 \\pm 0.023', '0.701 \\pm 0.046'),
    'XGBoost': ('0.891 \\pm 0.024', '0.674 \\pm 0.060'),
    'GCN': ('0.915 \\pm 0.018', '0.691 \\pm 0.072'),
    'GAT': ('0.906 \\pm 0.017', '0.681 \\pm 0.077'),
    'GIN': ('0.923 \\pm 0.017', '0.728 \\pm 0.058'),
    'SAGE': ('0.912 \\pm 0.019', '0.700 \\pm 0.072'),
    'MV-GNN$^{cross}$': ('0.907 \\pm 0.026', '0.697 \\pm 0.062'),
    'Graphormer': ('0.880 \\pm 0.027', '0.607 \\pm 0.075'),
    'GraphGPS': ('0.914 \\pm 0.017', '0.707 \\pm 0.043'),
    'Consensus-Classical': ('0.903 \\pm 0.024', '0.703 \\pm 0.050'),
    'Consensus-GNN': ('0.924 \\pm 0.016', '0.737 \\pm 0.075'),
    'Consensus-GT': ('0.916 \\pm 0.019', '0.707 \\pm 0.059'),
    'Consensus (Full)': ('0.932 \\pm 0.016', '0.772 \\pm 0.051'),
}

# 1__editing.tex writes this row's label plainly; the table prints it with the
# superscript. Lookups go through the plain form, output uses the pretty one.
LOOKUP = {'MV-GNN$^{cross}$': 'MV-GNN-cross'}

n_test = N_TOTAL / 10.0
n_train = 0.8 * N_TOTAL


def nd(c):
    return 2 if c in PCT else 3


def stat(vals):
    v = np.asarray(vals, float)
    n = len(v)
    sd = v.std(ddof=1)
    half = stats.t.ppf(0.975, df=n - 1) * np.sqrt((1.0 / n + n_test / n_train) * sd ** 2)
    return float(v.mean()), float(half)


def plain(label):
    """The label as 1__editing.tex writes it, for looking up its published means."""
    return LOOKUP.get(label, label)


def published_means():
    """The means exactly as 1__editing.tex prints them, keyed (label, cpt, col)."""
    tex = re.search(r'\\label\{tab:mmp_up_loo\}.*?\\end\{table\*\}',
                    open(os.path.join(HERE, '1__editing.tex')).read(), re.S).group(0)
    out, label = {}, None
    for line in tex.splitlines():
        if '&' not in line or 'multicolumn' in line or 'textbf{Model}' in line:
            continue
        fields = line.rstrip().rstrip('\\\\').split('&')
        if len(fields) != 10:
            continue
        head = fields[0].strip()
        if head.startswith('\\multirow'):
            # the label may itself contain braces (MV-GNN$^{cross}$), so take
            # everything after the \multirow prefix and drop the closing brace
            label = head[head.index('{*}{') + 4:].rstrip()[:-1]
        cpt_txt = fields[1].strip()
        if not re.fullmatch(r'0\.\d', cpt_txt):
            continue
        cpt = float(cpt_txt)
        for c, raw in zip(COLS, fields[2:]):
            v = re.sub(r'\\multirow\{2\}\{\*\}\{|\\underline\{|\{\\boldmath|\\textbf\{|[{}$]', '', raw).strip()
            if v:
                out[(label, cpt, c)] = v.split()[0]
    # ROC-AUC / PR-AUC are printed once in a \multirow on the 0.5 line; mirror
    # them onto the 0.8 key so every (label, cpt, col) lookup resolves.
    for (lab, cpt, c), v in list(out.items()):
        if c in FREE and cpt == 0.5:
            out[(lab, 0.8, c)] = v
    return out


def main():
    pub = published_means()
    cache, half = {}, {}
    for _, members in GROUPS:
        for key, model, label in members:
            if key not in cache:
                cache[key] = pd.read_csv(f'{SRC}/{key}_per_unit.csv')
            t = cache[key]
            t = t[(t.Split == 'Test') & (t.Model == model)]
            for cpt in CPTS:
                g = t[t.CPT == cpt]
                assert len(g) == 10, f'{label} cpt{cpt}: {len(g)} folds'
                for c in COLS:
                    m, h = stat(g[c].values)
                    half[(label, cpt, c)] = h
                    if c in FREE:        # four-decimal column: must match exactly
                        assert f'{m:.3f}' == pub[(plain(label), cpt, c)], (
                            f'{label} {c}: recomputed {m:.3f} vs published '
                            f'{pub[(plain(label), cpt, c)]}')

    # the published AUC intervals must also come back out unchanged
    for _, members in GROUPS:
        for _, _, label in members:
            for c, p_ in zip(('ROC-AUC', 'PR-AUC'), PUBLISHED[label]):
                got = f"{pub[(plain(label), 0.5, c)]} \\pm {half[(label, 0.5, c)]:.3f}"
                assert got == p_, f'{label} {c}: "{got}" vs published "{p_}"'
    print('all 28 published ROC-AUC / PR-AUC values and intervals reproduced exactly')

    def val(label, cpt, c):
        return float(pub[(plain(label), cpt, c)])

    rank = {}
    for c in COLS:
        pool = {val(l, cpt, c)
                for _, mem in GROUPS for _, _, l in mem
                for cpt in ((0.5,) if c in FREE else CPTS)}
        s_ = sorted(pool, reverse=True)
        rank[c] = (s_[0], s_[1] if len(s_) > 1 else None)

    def fmt(label, cpt, c):
        s_ = f'${pub[(plain(label), cpt, c)]} \\pm {half[(label, cpt, c)]:.{nd(c)}f}$'
        best, second = rank[c]
        v = val(label, cpt, c)
        if v == best:
            s_ = '{\\boldmath' + s_ + '}'
        elif v == second:
            s_ = f'\\underline{{{s_}}}'
        return s_

    L = ['\\begin{table*}[t]', '\\centering',
         '\\caption{Performance on the MMP summary assay (AID 720637) under 10-fold '
         'cross-validation. In each of ten rounds, one fold was the test set, the next fold '
         'the validation set, and the remaining eight folds the training set, with actives '
         'up-sampled to a 1:1 ratio. Values are the mean of the ten test-fold scores with '
         'Nadeau--Bengio corrected 95\\% confidence intervals; on the percentage columns the '
         'intervals are in percentage points. ROC-AUC and PR-AUC do not depend on the cutoff, '
         'so their values span both rows. Percentages are given to two decimals and all other '
         'values to three, with the best value per column in bold and the second best '
         'underlined.}', '',
         '\\label{tab:mmp_up_loo}', '\\resizebox{\\textwidth}{!}{%',
         '\\begin{tabular}{llcccccccc}', '\\toprule',
         '\\textbf{Model} & \\textbf{CPT} & \\textbf{Sensitivity} & \\textbf{Specificity} & '
         '\\textbf{CCR} & \\textbf{ROC-AUC (95\\% CI)} & \\textbf{PR-AUC (95\\% CI)} & '
         '\\textbf{MCC} & \\textbf{Bal-Acc} & \\textbf{F1} \\\\',
         ' & & (\\%) & (\\%) & (\\%) & & & & (\\%) & \\\\']
    for gname, members in GROUPS:
        L += ['\\midrule', f'\\multicolumn{{10}}{{l}}{{\\textit{{{gname}}}}} \\\\', '\\midrule']
        for _, _, label in members:
            for cpt in CPTS:
                row = [f'\\multirow{{2}}{{*}}{{{label}}}' if cpt == 0.5 else '', f'{cpt}']
                for c in COLS:
                    if c in FREE:
                        row.append(f'\\multirow{{2}}{{*}}{{{fmt(label, 0.5, c)}}}'
                                   if cpt == 0.5 else '')
                    else:
                        row.append(fmt(label, cpt, c))
                L.append(' & '.join(row) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}%', '}', '\\end{table*}', '']

    out = os.path.join(HERE, 'results', 'table_mmp_up_loo_allci.tex')
    with open(out, 'w') as fh:
        fh.write('\n'.join(L))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
