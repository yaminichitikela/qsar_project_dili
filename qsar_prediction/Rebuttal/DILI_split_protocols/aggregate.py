"""Turns the per-unit runs into summary tables and the comparison with the
current DILI results.

WHAT THIS COMPUTES
------------------
 1. The consensus ensembles per (protocol, stage, unit), on Held-223 and on the
    internal test, from the members' saved probabilities. Definitions are the
    ones in DILI_assay_link/aggregate_link_results.py:
        Consensus-Classical  RF, SVM, XGB
        Consensus-GNN        GCN, GAT, GIN, SAGE
        Consensus-GT         Graphormer, GraphGPS
        Consensus            the nine above
        Consensus-MVGNN      the nine above + MV-GNN
 2. mean +/- SD over units, per protocol / stage / model / cutoff, for
    Held-223 (primary) and the internal test (secondary).
 3. For CV, also the fold ensemble: the 10 fold models' Held-223 probabilities
    averaged into one prediction per family (one number, no SD).
 4. The comparison against the current DILI table (baseline/baseline_per_repeat
    .csv = DILI_assay_link Structure-only arm, 5 repeats, same Held-223):
    difference in means, Welch t-test and Mann-Whitney U. The baseline repeats
    and the new units come from different splits, so the samples are unpaired.
    The downsampling effect (random_repeat vs random_repeat_nodown) IS paired --
    the same 5 splits -- so it gets a paired t-test.
    With 5 units per arm these tests have little power; read the deltas and SDs
    first, the p-values second.
 5. LaTeX tables in the format of ../tables_dili_repeated_split.tex.

Usage: python aggregate.py
Outputs (results/):
    per_unit_all.csv, summary.csv, cv_fold_ensemble.csv,
    comparison_vs_baseline.csv, downsampling_effect.csv,
    tables_dili_split_protocols.tex
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
from common import (BASELINE, CLASSICAL, CPTS, GNN, GT, MVGNN, PROTOCOLS, STAGES,
                    data_dir, unit_dir)
from eval_utils import evaluate

SMOKE = os.environ.get('DILI_SMOKE') == '1'
RES = os.path.join(HERE, 'results_smoke' if SMOKE else 'results')
METRICS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'F1']
COMPARE = ['ROC-AUC', 'PR-AUC', 'CCR', 'MCC']
NINE = CLASSICAL + GNN + GT
ORDER = NINE + [MVGNN, 'Consensus-Classical', 'Consensus-GNN', 'Consensus-GT',
                'Consensus', 'Consensus-MVGNN']
PROTO_LABEL = {
    'random_repeat': 'repeated random 80/10/10 splits (5 repeats, 1:1 downsampled training)',
    'random_repeat_nodown': 'repeated random 80/10/10 splits (5 repeats, no downsampling)',
    'cv': '10-fold cross-validation (1:1 downsampled training)',
    'scaffold': 'repeated scaffold 8:1:1 splits (5 repeats, 1:1 downsampled training)',
    'random_repeat_up': 'repeated random 80/10/10 splits (5 repeats, 1:1 up-sampled training)',
    'cv_up': '10-fold cross-validation (1:1 up-sampled training)',
    'scaffold_up': 'repeated scaffold 8:1:1 splits (5 repeats, 1:1 up-sampled training)',
}


def ensembles(got):
    ens = {}
    if all(f in got for f in CLASSICAL):
        ens['Consensus-Classical'] = np.mean([got[f] for f in CLASSICAL], axis=0)
    if all(f in got for f in GNN):
        ens['Consensus-GNN'] = np.mean([got[f] for f in GNN], axis=0)
    if all(f in got for f in GT):
        ens['Consensus-GT'] = np.mean([got[f] for f in GT], axis=0)
    if all(f in got for f in NINE):
        ens['Consensus'] = np.mean([got[f] for f in NINE], axis=0)
        if MVGNN in got:
            ens['Consensus-MVGNN'] = np.mean([got[f] for f in NINE + [MVGNN]], axis=0)
    return ens


def load():
    ddir = data_dir()
    y_test = np.load(f'{ddir}/y_test.npy').astype(int)
    y_clin = np.load(f'{ddir}/y_clin.npy').astype(int)

    rows = [pd.read_csv(f) for f in glob.glob(f'{RES}/*/*_s[12]_u*_metrics.csv')]
    if not rows:
        print(f'no per-unit metrics under {RES}')
        sys.exit(1)
    per = pd.concat(rows, ignore_index=True)

    cons, cv_ens = [], []
    for proto, n_units in PROTOCOLS.items():
        for stage in STAGES:
            fold_probs = {}
            for u in range(n_units):
                y_it = y_clin[np.load(f'{unit_dir(proto, u)}/idx_itest.npy')]
                for split, y in (('test', y_test), ('itest', y_it)):
                    got = {}
                    for fam in NINE + [MVGNN]:
                        p = f'{RES}/{proto}/probs/{fam}_s{stage}_u{u}_{split}.npy'
                        if os.path.exists(p):
                            got[fam] = np.load(p)
                    if SMOKE and split == 'itest':
                        continue        # smoke runs score a random itest subset
                    if split == 'test':
                        for fam, pr in got.items():
                            fold_probs.setdefault(fam, []).append(pr)
                    for name, pr in ensembles(got).items():
                        if split == 'test':
                            fold_probs.setdefault(name, []).append(pr)
                        for cpt in CPTS:
                            cons.append({**evaluate(y, pr, name, cpt), 'Protocol': proto,
                                         'Stage': stage, 'Unit': u, 'Split': split})
            if proto == 'cv':
                for name, lst in fold_probs.items():
                    if len(lst) == n_units:
                        for cpt in CPTS:
                            cv_ens.append({**evaluate(y_test, np.mean(lst, axis=0), name, cpt),
                                           'Stage': stage, 'n_folds': n_units})
    allrows = pd.concat([per, pd.DataFrame(cons)], ignore_index=True)
    return allrows, pd.DataFrame(cv_ens)


def summarize(df, keys):
    out = []
    for k, g in df.groupby(keys):
        row = dict(zip(keys, k), n_units=g.Unit.nunique())
        for m in METRICS:
            row[f'{m}_mean'] = round(float(g[m].mean()), 4)
            row[f'{m}_sd'] = round(float(g[m].std(ddof=1)) if len(g) > 1 else 0.0, 4)
        out.append(row)
    return pd.DataFrame(out)


def compare(test, base):
    out = []
    for proto in PROTOCOLS:
        for stage in STAGES:
            for model in ORDER:
                a = base[(base.Stage == stage) & (base.Model == model) & (base.CPT == 0.5)]
                b = test[(test.Protocol == proto) & (test.Stage == stage)
                         & (test.Model == model) & (test.CPT == 0.5)]
                if len(a) < 2 or len(b) < 2:
                    continue
                row = dict(protocol=proto, stage=stage, model=model,
                           n_baseline=len(a), n_protocol=len(b))
                for m in COMPARE:
                    x, z = a[m].values, b[m].values
                    row[f'{m}_baseline'] = round(x.mean(), 4)
                    row[f'{m}_protocol'] = round(z.mean(), 4)
                    row[f'{m}_delta'] = round(z.mean() - x.mean(), 4)
                    row[f'{m}_welch_p'] = round(float(stats.ttest_ind(z, x, equal_var=False).pvalue), 4) \
                        if np.std(np.r_[x, z]) > 0 else 1.0
                    try:
                        row[f'{m}_mwu_p'] = round(float(stats.mannwhitneyu(z, x).pvalue), 4)
                    except ValueError:
                        row[f'{m}_mwu_p'] = 1.0
                out.append(row)
    return pd.DataFrame(out)


def downsampling_effect(test):
    out = []
    for stage in STAGES:
        for model in ORDER:
            sel = lambda p: test[(test.Protocol == p) & (test.Stage == stage) & (test.Model == model)
                                 & (test.CPT == 0.5)].set_index('Unit')
            d, n = sel('random_repeat'), sel('random_repeat_nodown')
            shared = sorted(set(d.index) & set(n.index))
            if len(shared) < 2:
                continue
            row = dict(stage=stage, model=model, n_units=len(shared))
            for m in COMPARE + ['Sensitivity', 'Specificity']:
                x, z = n.loc[shared, m].values, d.loc[shared, m].values
                diff = z - x
                row[f'{m}_nodown'] = round(x.mean(), 4)
                row[f'{m}_down'] = round(z.mean(), 4)
                row[f'{m}_delta'] = round(diff.mean(), 4)
                row[f'{m}_paired_p'] = round(float(stats.ttest_rel(z, x).pvalue), 4) \
                    if np.std(diff) > 0 else 1.0
            out.append(row)
    return pd.DataFrame(out)


def upsampling_effect(test):
    """Up-sampled vs 1:1 downsampled, paired on the shared splits.

    Each *_up protocol reuses its downsampled counterpart's splits (same seeds,
    identical validation and internal-test sets), so unit u of the two arms is
    the same split and the per-unit differences are paired."""
    from common import UP_PAIRS
    out = []
    for up, down in UP_PAIRS.items():
        for stage in STAGES:
            for model in ORDER:
                sel = lambda p: test[(test.Protocol == p) & (test.Stage == stage)
                                     & (test.Model == model)
                                     & (test.CPT == 0.5)].set_index('Unit')
                u, d = sel(up), sel(down)
                shared = sorted(set(u.index) & set(d.index))
                if len(shared) < 2:
                    continue
                row = dict(protocol=down, stage=stage, model=model, n_units=len(shared))
                for m in COMPARE + ['Sensitivity', 'Specificity', 'F1']:
                    x, z = d.loc[shared, m].values, u.loc[shared, m].values
                    diff = z - x
                    row[f'{m}_down'] = round(x.mean(), 4)
                    row[f'{m}_up'] = round(z.mean(), 4)
                    row[f'{m}_delta'] = round(diff.mean(), 4)
                    row[f'{m}_paired_p'] = round(float(stats.ttest_rel(z, x).pvalue), 4) \
                        if np.std(diff) > 0 else 1.0
                out.append(row)
    return pd.DataFrame(out)


# ── LaTeX, in the format of tables_dili_repeated_split.tex ──────────────────
GROUPS = [('Classical ML', CLASSICAL), ('Message-passing GNN', GNN),
          ('Multi-view GNN', [MVGNN]), ('Graph transformer', GT),
          ('Consensus', ['Consensus-Classical', 'Consensus-GNN', 'Consensus-GT',
                         'Consensus', 'Consensus-MVGNN'])]
TEX_NAME = {'Consensus-Classical': r'Cons.\ Classical', 'Consensus-GNN': r'Cons.\ GNN',
            'Consensus-GT': r'Cons.\ GT', 'Consensus-MVGNN': r'Cons.\ + MVGNN'}
STAGE_NAME = {1: 'Stage 1 (Proxy-only)', 2: 'Stage 2 (Proxy+DILI)'}


def fmt(m, mean, sd):
    return f'{mean:.2f} $\\pm$ {sd:.2f}' if m in ('Sensitivity', 'Specificity', 'CCR') \
        else f'{mean:.4f} $\\pm$ {sd:.4f}'


def tex_table(summ, proto, stage, cpt):
    s = summ[(summ.Protocol == proto) & (summ.Stage == stage) & (summ.CPT == cpt)
             & (summ.Split == 'test')].set_index('Model')
    if s.empty:
        return ''
    best = {}
    for m in METRICS:
        # best mean; ties at display precision broken by the smaller SD
        nd = 2 if m in ('Sensitivity', 'Specificity', 'CCR') else 4
        key = s[[f'{m}_mean', f'{m}_sd']].copy()
        key['r'] = key[f'{m}_mean'].round(nd)
        best[m] = key.sort_values(['r', f'{m}_sd'], ascending=[False, True]).index[0]
    lab = f'tab:dili_{proto}_s{stage}_cpt{int(cpt * 100)}'
    L = [r'\begin{table}[htbp]', r'\centering', r'\footnotesize',
         r'\setlength{\tabcolsep}{3.5pt}',
         rf'\caption{{DILI prediction performance, {STAGE_NAME[stage]}, '
         rf'{PROTO_LABEL[proto]}. Mean $\pm$ standard deviation over '
         rf'{int(s.n_units.max())} units on the fixed Held-223 external test set; '
         rf'classification cutoff {cpt:.2f}. Sensitivity, specificity and CCR are '
         rf'percentages. Best value per column in bold; where means tie at the '
         rf'precision shown, the smaller standard deviation is taken as better.}}',
         rf'\label{{{lab}}}', r'\begin{tabular}{lccccccc}', r'\hline',
         r'Model & Sensitivity & Specificity & CCR & ROC-AUC & PR-AUC & MCC & F1 \\',
         r'\hline']
    for gname, members in GROUPS:
        present = [x for x in members if x in s.index]
        if not present:
            continue
        L.append(rf'\multicolumn{{8}}{{l}}{{\textit{{{gname}}}}} \\')
        for mod in present:
            cells = []
            for m in METRICS:
                c = fmt(m, s.loc[mod, f'{m}_mean'], s.loc[mod, f'{m}_sd'])
                cells.append(rf'\textbf{{{c}}}' if best[m] == mod else c)
            L.append(rf'\quad {TEX_NAME.get(mod, mod)} & ' + ' & '.join(cells) + r' \\')
    L += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(L)


def tex_comparison(comp, stage):
    if comp.empty:
        return ''
    c = comp[comp.stage == stage]
    if c.empty:
        return ''
    protos = [p for p in PROTOCOLS if p in set(c.protocol)]
    L = [r'\begin{table}[htbp]', r'\centering', r'\footnotesize',
         r'\setlength{\tabcolsep}{3pt}',
         rf'\caption{{Change in Held-223 ROC-AUC against the current DILI results '
         rf'(repeated random train/validation splits, no downsampling), '
         rf'{STAGE_NAME[stage]}, cutoff 0.50. Positive = the protocol is better. '
         rf'$^*$ Welch $p<0.05$.}}',
         rf'\label{{tab:dili_protocols_vs_baseline_s{stage}}}',
         r'\begin{tabular}{l' + 'c' * (len(protos) + 1) + '}', r'\hline',
         'Model & Baseline & ' + ' & '.join(p.replace('_', r'\_') for p in protos) + r' \\',
         r'\hline']
    for mod in ORDER:
        r = c[c.model == mod].set_index('protocol')
        if r.empty:
            continue
        base = r['ROC-AUC_baseline'].iloc[0]
        cells = []
        for p in protos:
            if p in r.index:
                d, pv = r.loc[p, 'ROC-AUC_delta'], r.loc[p, 'ROC-AUC_welch_p']
                cells.append(f'{d:+.4f}' + ('$^*$' if pv < 0.05 else ''))
            else:
                cells.append('--')
        L.append(f'{TEX_NAME.get(mod, mod)} & {base:.4f} & ' + ' & '.join(cells) + r' \\')
    L += [r'\hline', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(L)


def main():
    allrows, cv_ens = load()
    allrows.to_csv(f'{RES}/per_unit_all.csv', index=False)
    test = allrows[allrows.Split == 'test']
    summ = summarize(allrows[allrows.Split.isin(['test', 'itest'])],
                     ['Protocol', 'Split', 'Stage', 'Model', 'CPT'])
    summ.to_csv(f'{RES}/summary.csv', index=False)
    if len(cv_ens):
        cv_ens.to_csv(f'{RES}/cv_fold_ensemble.csv', index=False)

    base = pd.read_csv(f'{BASELINE}/baseline_per_repeat.csv')
    base = base[base.Split.str.lower() == 'test']
    comp = compare(test, base)
    comp.to_csv(f'{RES}/comparison_vs_baseline.csv', index=False)
    ds = downsampling_effect(test)
    ds.to_csv(f'{RES}/downsampling_effect.csv', index=False)
    us = upsampling_effect(test)
    us.to_csv(f'{RES}/upsampling_effect.csv', index=False)

    tex = ['% DILI split protocols (random repeat, 10-fold CV, scaffold), '
           'corrected graph transformers.',
           '% Structure-only; Held-223 is the fixed test set; generated by '
           'DILI_split_protocols/aggregate.py', '']
    for proto in PROTOCOLS:
        for cpt in CPTS:
            for stage in STAGES:
                tex.append(tex_table(summ, proto, stage, cpt))
    for stage in STAGES:
        tex.append(tex_comparison(comp, stage))
    with open(f'{RES}/tables_dili_split_protocols.tex', 'w') as fh:
        fh.write('\n'.join(tex))

    # ── console headline ────────────────────────────────────────────────────
    pd.set_option('display.width', 200)
    print(f'{len(test)} Held-223 rows; units per protocol: '
          + ', '.join(f'{p}={test[test.Protocol == p].Unit.nunique()}' for p in PROTOCOLS))
    if len(comp):
        print('\nHeld-223 ROC-AUC vs current DILI results (cutoff 0.5):')
        show = comp[['protocol', 'stage', 'model', 'ROC-AUC_baseline', 'ROC-AUC_protocol',
                     'ROC-AUC_delta', 'ROC-AUC_welch_p', 'CCR_delta']]
        print(show.to_string(index=False))
        sig = comp[(comp['ROC-AUC_welch_p'] < 0.05)]
        print(f'\n{len(sig)} of {len(comp)} protocol/model comparisons reach Welch p < 0.05 '
              f'on ROC-AUC (5 units per arm, so the test is weak).')
    if len(us):
        print('\nUP-SAMPLED vs 1:1 DOWNSAMPLED training (paired on the same splits, '
              'Held-223 ROC-AUC, cutoff 0.5):')
        print(us[['protocol', 'stage', 'model', 'ROC-AUC_down', 'ROC-AUC_up',
                  'ROC-AUC_delta', 'ROC-AUC_paired_p']].to_string(index=False))
        sig = us[us['ROC-AUC_paired_p'] < 0.05]
        print(f'{len(sig)} of {len(us)} comparisons reach paired p < 0.05 on ROC-AUC.')

    print(f'\n-> {RES}/summary.csv, comparison_vs_baseline.csv, downsampling_effect.csv,')
    print(f'   cv_fold_ensemble.csv, tables_dili_split_protocols.tex')


if __name__ == '__main__':
    main()
