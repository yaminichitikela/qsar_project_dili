r"""DILI split-protocol tables in the same format as the assay tables
(../tables_hepg2_updated.tex, ../tables_chok1_updated.tex).

Matches the assay format:
  * booktabs table*, scaled to \textwidth, (%) unit row under the header
  * columns Sensitivity, Specificity, CCR, ROC-AUC, PR-AUC, MCC, Bal-Acc, F1
  * percentages to 1 decimal, all other values to 3 decimals
  * the +/- interval on all eight metrics (percentage columns in percentage points)
  * random repeated split: interval = SD across the 5 repeats
    10-fold CV and scaffold: interval = Nadeau--Bengio corrected 95% half-width,
        t(0.975, n-1) * sqrt((1/n + n_test/n_train) * var)
    the same formula as compute_ci_significance.corrected_ci and
    build_upsampling_1to5_tex.corrected_ci. Here n_test is the Held-223 test set
    (222 compounds) and n_train is the mean training-set size of the stage.
  * groups and names as in the assay tables; MV-GNN is a baseline and is not
    part of any consensus (Consensus (Full) = the nine other models)
  * bold = best displayed value per column, within each stage

DILI-specific: one cutoff per table (default 0.50) and Stage 1 (Proxy-only)
and Stage 2 (Proxy+DILI) stacked in the same table, one table per split.

Usage: python make_assay_style_tables.py [down|up] [cutoff]
Output: results/tables_dili_split_protocols_assay_style_cpt<NN>.tex
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import data_dir, unit_dir

SD_TEXT = ('Intervals on every metric are the standard deviation across repeats; '
           'on the percentage columns they are in percentage points.')
PROTO_SETS = {
    'down': [
        ('random_repeat', 5, 'repeated 80/10/10 split (mean $\\pm$ std across 5 repeats), '
         '1:1 downsampled train', 'sd', SD_TEXT, 'random'),
        ('cv', 10, '10-fold cross-validation, 1:1 downsampled train (mean across 10 folds)',
         'nb', None, 'cv'),
        ('scaffold', 5, 'scaffold split, 5 repeats, 1:1 downsampled train (mean across 5 repeats)',
         'nb', None, 'scaffold'),
    ],
    'up': [
        ('random_repeat_up', 5, 'repeated 80/10/10 split (mean $\\pm$ std across 5 repeats), '
         '1:1 up-sampled train', 'sd', SD_TEXT, 'random_up'),
        ('cv_up', 10, '10-fold cross-validation, 1:1 up-sampled train (mean across 10 folds)',
         'nb', None, 'cv_up'),
        ('scaffold_up', 5, 'scaffold split, 5 repeats, 1:1 up-sampled train (mean across 5 repeats)',
         'nb', None, 'scaffold_up'),
    ],
}
UP_NOTE = ('Training sets are balanced by up-sampling: DILI-positive compounds are duplicated '
           'with replacement up to the negative count, keeping every negative. The validation '
           'and internal-test splits keep their natural class ratio, and Held-223 is never '
           'resampled. The splits are the same ones used for the 1:1 downsampled tables.')
NB_TEXT = ('Intervals on every metric are Nadeau--Bengio corrected 95\\% intervals '
           '(correction $1/n + n_{\\mathrm{test}}/n_{\\mathrm{train}}$, with $n_{\\mathrm{test}}$ '
           'the 222 Held-223 compounds and $n_{\\mathrm{train}}$ the mean training-set size); '
           'on the percentage columns they are in percentage points.')
XU_KEY = 'XU'
XU_CLF = 'SVM'            # the classifier Xu et al.'s own selection procedure picks
# The TX2017 runs are balanced the same way as the table they appear in: the
# up-sampled tables read the over-sampled run, the downsampled tables read the
# run that reuses this folder's 1:1 downsampled idx_train.npy. Both address the
# same split indices, so units line up one-to-one either way.
TX_ROOT = os.path.join(os.path.dirname(HERE), 'DILI_tx2017_splits')
XU_SRC = {'up': os.path.join(TX_ROOT, 'results', 'tx_per_unit_all.csv'),
          'down': os.path.join(TX_ROOT, 'results_down', 'tx_per_unit_all.csv')}
XU_PROTO = {'random_repeat_up': 'random_repeat', 'cv_up': 'cv', 'scaffold_up': 'scaffold',
            'random_repeat': 'random_repeat', 'cv': 'cv', 'scaffold': 'scaffold'}
XU_GROUP = ("Xu et al. ``Structure only'' model (replicated)",
            [(XU_KEY, 'Structure only$^{\\ddagger}$')])
_XU_NOTE_HEAD = ("$^{\\ddagger}$``Structure only'' is the feature setting of that name in "
                 'Xu et al., replicated here on the same splits: ECFP4 at 1024 bits, '
                 'Random-Forest Gini feature selection (top 180 bits, re-selected inside each '
                 'unit), ')
_XU_NOTE_TAIL = ('and an SVM classifier -- the combination their selection procedure chooses on '
                 'these data, in 37 of the 40 unit/stage fits. Because that workflow is '
                 'replicated as published, it is not hyperparameter-matched to the classical '
                 'models above: those use all 2048 fingerprint bits with per-family Optuna '
                 'tuning and balanced class weights, so the rows are not a like-for-like '
                 'comparison of the same algorithm. It is reported for comparison and is not '
                 'part of any consensus.')
XU_NOTE = {
    'up': _XU_NOTE_HEAD + 'random over-sampling of the training set ' + _XU_NOTE_TAIL,
    'down': _XU_NOTE_HEAD + ('the same 1:1 downsampled training rows as the models above, '
                             'reused unchanged, ') + _XU_NOTE_TAIL,
}


def load_xu(proto, cpt):
    """Per-unit Xu et al. rows for the protocol this table reports, or None.

    The TX2017 runs use the same split indices and the same balancing scheme as
    this table, so their units line up one-to-one with it.
    """
    src = XU_PROTO.get(proto)
    path = XU_SRC['up' if proto.endswith('_up') else 'down']
    if src is None or not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[(d.Protocol == src) & (d.Model == XU_CLF) & (d.Split == 'test')
          & ((d.CPT - cpt).abs() < 1e-9)].copy()
    if d.empty:
        return None
    d['Model'] = XU_KEY
    return d


GROUPS = [
    ('Classical Machine Learning', [('RF', 'Random Forest'), ('SVM', 'SVM'), ('XGB', 'XGBoost')]),
    ('Graph Neural Networks', [('GCN', 'GCN'), ('GAT', 'GAT'), ('GIN', 'GIN'), ('SAGE', 'SAGE')]),
    ('Graph Transformers', [('Graphormer', 'Graphormer$^{\\dagger}$'),
                            ('GraphGPS', 'GraphGPS$^{\\dagger}$')]),
    ('Multi-View GNN (baseline)', [('MVGNN', 'MV-GNN$^{\\text{cross}}$')]),
    ('Consensus Ensembles', [('Consensus-Classical', 'Consensus-Classical'),
                             ('Consensus-GNN', 'Consensus-GNN'),
                             ('Consensus-GT', 'Consensus-GT$^{\\dagger}$'),
                             ('Consensus', 'Consensus (Full)$^{\\dagger}$')]),
]
COLS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
PCT = {'Sensitivity', 'Specificity', 'CCR', 'Bal-Acc'}
# Every metric carries its own interval. The threshold-dependent columns
# (Sensitivity, Specificity, CCR, MCC, Bal-Acc, F1) vary across units exactly as
# the AUCs do, so the same convention applies to all eight.
CI_COLS = set(COLS)
STAGES = [(1, 'Stage 1 (Proxy-only)'), (2, 'Stage 2 (Proxy+DILI)')]
FOOT = ('\\begin{flushleft}\\footnotesize $^{\\dagger}$Corrected graph-transformer implementation '
        '(true Graphormer with global all-pairs attention and spatial/edge encodings; GraphGPS with '
        'sign-flipped Laplacian positional encodings). MV-GNN$^{\\text{cross}}$ is reported as a '
        'baseline comparison architecture and is not included in any consensus ensemble.'
        '\\end{flushleft}')


def nd(c):
    """Displayed decimals: percentages to two, every other value to three."""
    return 2 if c in PCT else 3


def n_train(proto, n_units, stage, n_ncv):
    tr = [len(np.load(f'{unit_dir(proto, u)}/idx_train.npy')) for u in range(n_units)]
    return float(np.mean(tr)) + (n_ncv if stage == 2 else 0)


def interval(vals, kind, ntr, nte):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    sd = vals.std(ddof=1)
    if kind == 'sd':
        return sd
    se = np.sqrt((1.0 / n + nte / ntr) * sd ** 2)
    return stats.t.ppf(0.975, df=n - 1) * se


def table(per, proto, n_units, desc, kind, int_text, tag, cpt, n_ncv, n_test):
    L = ['\\begin{table*}[t]', '\\centering',
         f'\\caption{{Performance on the FDA-approved DILI held-out test set (Held-223; 222 compounds, '
         f'155 hepatotoxic/67 non-hepatotoxic) -- {desc}; Stage 1 (Proxy-only) and Stage 2 '
         f'(Proxy+DILI); classification cutoff {cpt:.1f}. Rows marked $^{{\\dagger}}$ use the '
         f'corrected graph-transformer implementations. {int_text or NB_TEXT} '
         f'{UP_NOTE + " " if proto.endswith("_up") else ""}'
         f'Percentages are given to two decimals and all other values to three. '
         f'Best value per column within each stage in bold, second best underlined.}}',
         f'\\label{{tab:dili_{tag}_assay_style}}',
         '\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{lcccccccc}', '\\toprule',
         '\\textbf{Model} & \\textbf{Sensitivity} & \\textbf{Specificity} & \\textbf{CCR} & '
         '\\textbf{ROC-AUC} & \\textbf{PR-AUC} & \\textbf{MCC} & \\textbf{Bal-Acc} & \\textbf{F1} \\\\',
         ' & (\\%) & (\\%) & (\\%) & & & & (\\%) & \\\\']
    xu = load_xu(proto, cpt)
    groups = list(GROUPS)
    if xu is not None:
        # inserted immediately before the graph transformers, as a comparison arm
        at = next(i for i, (g, _) in enumerate(groups) if g == 'Graph Transformers')
        groups.insert(at, XU_GROUP)

    for stage, sname in STAGES:
        sub = per[(per.Protocol == proto) & (per.Stage == stage)]
        if xu is not None:
            sub = pd.concat([sub, xu[xu.Stage == stage]], ignore_index=True)
        ntr = n_train(proto, n_units, stage, n_ncv)
        cells = {}
        for _, members in groups:
            for key, _ in members:
                g = sub[sub.Model == key]
                assert g.Unit.nunique() == n_units, f'{proto} s{stage} {key}: {g.Unit.nunique()} units'
                cells[key] = {c: round(float(g[c].mean()), nd(c)) for c in COLS}
                for c in CI_COLS:
                    cells[key][c + '_ci'] = interval(g[c].values, kind, ntr, n_test)
        # Ranked on the DISPLAYED value, so the marks match what the reader sees.
        # Ties share a rank: every cell equal to the best is bold, every cell
        # equal to the next distinct value is underlined.
        ranked = {c: sorted({v[c] for v in cells.values()}, reverse=True) for c in COLS}
        best = {c: r[0] for c, r in ranked.items()}
        second = {c: (r[1] if len(r) > 1 else None) for c, r in ranked.items()}
        L += ['\\midrule', f'\\multicolumn{{9}}{{c}}{{\\textbf{{{sname}}}}} \\\\']
        for gname, members in groups:
            L += ['\\midrule', f'\\multicolumn{{9}}{{l}}{{\\textit{{{gname}}}}} \\\\', '\\midrule']
            for key, label in members:
                row = []
                for c in COLS:
                    v = cells[key][c]
                    s = (f'${v:.{nd(c)}f} \\pm {cells[key][c + "_ci"]:.{nd(c)}f}$'
                         if c in CI_COLS else f'{v:.{nd(c)}f}')
                    if v == best[c]:
                        # \textbf does not bold math mode; \boldmath does (incl. the \pm)
                        s = f'\\textbf{{\\boldmath{s}}}' if c in CI_COLS else f'\\textbf{{{s}}}'
                    elif v == second[c]:
                        s = f'\\underline{{{s}}}'
                    row.append(s)
                L.append(f'{label} & ' + ' & '.join(row) + ' \\\\')
    foot = FOOT if xu is None else FOOT.replace(
        '\\end{flushleft}',
        ' ' + XU_NOTE['up' if proto.endswith('_up') else 'down'] + '\\end{flushleft}')
    L += ['\\bottomrule', '\\end{tabular}%', '}', foot, '\\end{table*}', '']
    return '\n'.join(L)


def main():
    args = [a for a in sys.argv[1:]]
    which = 'down'
    if args and args[0] in PROTO_SETS:
        which = args.pop(0)
    cpt = float(args[0]) if args else 0.5
    per = pd.read_csv(os.path.join(HERE, 'results', 'per_unit_all.csv'))
    per = per[(per.Split == 'test') & ((per.CPT - cpt).abs() < 1e-9)]
    ddir = data_dir()
    n_ncv = len(np.load(f'{ddir}/y_ncv.npy'))
    n_test = len(np.load(f'{ddir}/y_test.npy'))
    out = [f'% DILI split protocols ({which}-sampled training) in the assay-table format, '
           f'cutoff {cpt:.2f}.',
           '% Generated by DILI_split_protocols/make_assay_style_tables.py; needs booktabs, graphicx, amsmath.',
           '']
    out += [table(per, p, n, d, k, t, tag, cpt, n_ncv, n_test)
            for p, n, d, k, t, tag in PROTO_SETS[which]]
    suffix = '' if which == 'down' else '_upsampled'
    path = os.path.join(HERE, 'results',
                        f'tables_dili_split_protocols_assay_style{suffix}'
                        f'_cpt{int(round(cpt * 100))}.tex')
    with open(path, 'w') as fh:
        fh.write('\n'.join(out))
    print(f'-> {path}')


if __name__ == '__main__':
    main()
