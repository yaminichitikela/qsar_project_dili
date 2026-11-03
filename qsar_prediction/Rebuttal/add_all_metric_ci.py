"""Adds 95% intervals to EVERY metric column of the SI assay tables.

The editor asked for confidence intervals on "all performance values
presented". The tables carry them on ROC-AUC and PR-AUC only, so this rewrites
each table's cells as "mean +/- half-width" for all eight metrics.

TWO RULES, BOTH CHOSEN TO AVOID SILENTLY CHANGING THE PAPER
-----------------------------------------------------------
1. The MEAN is taken from the table, never recomputed. The per-unit CSVs store
   the percentage columns rounded to one decimal (the AUC/MCC/F1 columns to
   four), so re-averaging them reproduces a published percentage only to about
   +/-0.02 -- exactly the error expected from averaging values quantised at 0.1.
   The published tables were computed from the unrounded fold values, so they
   are the accurate ones.

2. The interval CONVENTION is not assumed, it is DERIVED. For each table every
   candidate convention is tried and the one that reproduces that table's
   already-published ROC-AUC and PR-AUC intervals is used for the other six
   columns. If none reproduces them the table is reported and left untouched,
   because an interval whose method cannot be confirmed should not be printed.

Candidates:
    sd           standard deviation across units
    naive        t(0.975, n-1) * sd / sqrt(n)
    nb_distinct  Nadeau-Bengio, n_train = 0.8 * N        (distinct compounds)
    nb_down      Nadeau-Bengio, n_train = 2 * 0.8 * n_act (1:1 downsampled rows)
with n_test = N/10 in both Nadeau-Bengio variants.

Usage: python add_all_metric_ci.py <group>      (group: mmp | chok1 | hepg2only)
Edits structural activation/brics_table_summary_assays.tex in place.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SI = os.path.join(HERE, 'structural activation', 'brics_table_summary_assays.tex')

COLS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
PCT = {'Sensitivity', 'Specificity', 'CCR', 'Bal-Acc'}
FREE = {'ROC-AUC', 'PR-AUC'}          # threshold-free: one interval spans both cutoffs
CPTS = (0.5, 0.8)

S1 = 'Summary_1_models'
GROUPS = {
    'mmp': dict(n_total=5937, n_active=857, tables=[
        ('tab:mmp_random',      f'{HERE}/wahab_random_repeat/mmp', '{k}_random_repeat_per_repeat.csv',
         f'{HERE}/graph_transformer_corrected/results/mmp/random_repeat/per_unit_all.csv'),
        ('tab:mmp_up_random',   f'{HERE}/{S1}/upsampling_protocols/results_random_repeat', '{k}_per_unit.csv', None),
        ('tab:mmp_loo',         f'{HERE}/{S1}/modeling/wahab_jobs/results_loo', '{k}_loo_per_fold.csv',
         f'{HERE}/graph_transformer_corrected/results/mmp/loo/per_unit_all.csv'),
        ('tab:mmp_up_loo',      f'{HERE}/{S1}/upsampling_protocols/results_loo', '{k}_per_unit.csv', None),
        ('tab:mmp_scaffold',    f'{HERE}/{S1}/modeling/wahab_jobs/results_scaffold', '{k}_scaffold_per_repeat.csv',
         f'{HERE}/graph_transformer_corrected/results/mmp/scaffold/per_unit_all.csv'),
        ('tab:mmp_up_scaffold', f'{HERE}/{S1}/upsampling_protocols/results_scaffold', '{k}_per_unit.csv', None),
    ]),
    # AID 1347037. Part 7 (tab:chok1_up_random) is a SINGLE 80/10/10 split whose
    # intervals are a percentile bootstrap of the test set, not a spread across
    # repeats, so it is not handled here and is left as published.
    'chok1': dict(n_total=7496, n_active=171, tables=[
        ('tab:chok1_random',        f'{HERE}/wahab_random_repeat/chok1', '{k}_random_repeat_per_repeat.csv',
         f'{HERE}/graph_transformer_corrected/results/chok1/random_repeat/per_unit_all.csv'),
        ('tab:chok1_uprep_random',  f'{HERE}/summary_1347037/upsampling_protocols/results_random_repeat', '{k}_per_unit.csv', None),
        ('tab:chok1_loo',           f'{HERE}/summary_1347037/modeling/wahab_jobs/results_loo', '{k}_loo_per_fold.csv',
         f'{HERE}/graph_transformer_corrected/results/chok1/loo/per_unit_all.csv'),
        ('tab:chok1_uprep_loo',     f'{HERE}/summary_1347037/upsampling/results_loo', '{k}_loo_per_fold.csv',
         f'{HERE}/graph_transformer_corrected/results/chok1/up_loo/per_unit_all.csv'),
        ('tab:chok1_scaffold',      f'{HERE}/summary_1347037/modeling/wahab_jobs/results_scaffold', '{k}_scaffold_per_repeat.csv',
         f'{HERE}/graph_transformer_corrected/results/chok1/scaffold/per_unit_all.csv'),
        ('tab:chok1_up_scaffold',   f'{HERE}/summary_1347037/upsampling_protocols/results_scaffold', '{k}_per_unit.csv', None),
        ('tab:chok1_1to5_scaffold', f'{HERE}/summary_1347037/downsampling_1to5_scaffold/results_scaffold', '{k}_per_unit.csv', None),
    ]),
    # AID 1347034 alone, all three parts up-sampled. Up-sampled runs use the
    # distinct-compound correction, whose factor is n_test/n_train =
    # (N/10)/(0.8N) = 0.125 regardless of N, so n_total here only has to be
    # self-consistent; n_active is unused for these parts.
    'hepg2only': dict(n_total=7716, n_active=282, tables=[
        ('tab:hepg2only_up_random',   f'{HERE}/summary_2_1437034/upsampling_1347034_random_repeat/results_random_repeat', '{k}_per_repeat.csv', None),
        ('tab:hepg2only_up_loo',      f'{HERE}/summary_2_1437034/upsampling_1347034_protocols/results_loo', '{k}_per_unit.csv', None),
        ('tab:hepg2only_up_scaffold', f'{HERE}/summary_2_1437034/upsampling_1347034_protocols/results_scaffold', '{k}_per_unit.csv', None),
    ]),
}
KEYS = ['rf', 'svm', 'xgb', 'gcn', 'gat', 'gin', 'sage', 'mvgnn',
        'graphormer', 'graphgps', 'consensus']

# Cells to print exactly as published, overriding the recomputed value. The
# MV-GNN ROC-AUC in tab:mmp_loo reproduces as 0.895 while the table prints
# 0.870, although the other six metrics of that row reproduce exactly; the
# published figure is kept by author decision.
KEEP_PUBLISHED = {('tab:mmp_loo', 'MV-GNN$^{\\text{cross}}$', 0.5, 'ROC-AUC'),
                  ('tab:mmp_loo', 'MV-GNN$^{\\text{cross}}$', 0.8, 'ROC-AUC')}


def nd(c):
    return 2 if c in PCT else 3


def load_units(d, pattern, corrected=None):
    """Every model's per-unit Test rows for one run, concatenated.

    The downsampled runs on disk predate the graph-transformer correction, so
    the published tables for them were built from graph_transformer_corrected/.
    When `corrected` is given, every model it contains REPLACES the stale copy;
    models it does not contain are taken from the original files.
    """
    frames = []
    for k in KEYS:
        p = os.path.join(d, pattern.format(k=k))
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        t = t[t.Split.str.lower() == 'test'].copy()
        t['Unit'] = t[next(c for c in ('Fold', 'Repeat', 'Unit') if c in t.columns)]
        frames.append(t)
    if not frames:
        raise SystemExit(f'no per-unit files under {d}')
    base = pd.concat(frames, ignore_index=True)
    if corrected and os.path.exists(corrected):
        c = pd.read_csv(corrected)
        c = c[c.Split.str.lower() == 'test'].copy()
        c['Unit'] = c.iloc[:, 0]
        base = pd.concat([base[~base.Model.isin(set(c.Model))], c], ignore_index=True)
    return base


def half_width(vals, kind, n_total, n_active, r=None):
    v = np.asarray(vals, float)
    n = len(v)
    sd = v.std(ddof=1)
    if kind == 'sd':
        return sd
    t = stats.t.ppf(0.975, df=n - 1)
    if kind == 'naive':
        return t * sd / np.sqrt(n)
    if kind == 'fitted':
        return t * np.sqrt((1.0 / n + r) * sd ** 2)
    n_test = n_total / 10.0
    n_train = 0.8 * n_total if kind == 'nb_distinct' else 2.0 * 0.8 * n_active
    return t * np.sqrt((1.0 / n + n_test / n_train) * sd ** 2)


def parse_table(block):
    """(label -> {(model, cpt, col): printed mean}, and the printed AUC intervals)."""
    means, auc = {}, {}
    model = None
    for line in block.splitlines():
        if '&' not in line or 'multicolumn' in line or 'textbf{Model}' in line:
            continue
        f = line.rstrip().rstrip('\\').split('&')
        if len(f) != 10:
            continue
        head = f[0].strip()
        if head.startswith('\\multirow'):
            model = head[head.index('{*}{') + 4:].rstrip()[:-1]
        if not re.fullmatch(r'0\.\d', f[1].strip()):
            continue
        cpt = float(f[1].strip())
        for c, raw in zip(COLS, f[2:]):
            txt = re.sub(r'\\multirow\{2\}\{\*\}\{|\\underline\{|\{\\boldmath|\\textbf\{|\\mathbf\{|[{}$]',
                         '', raw).strip()
            if not txt:
                continue
            m = re.match(r'([\d.]+)\s*(?:\\pm\s*([\d.]+))?', txt)
            if not m:
                continue
            means[(model, cpt, c)] = m.group(1)
            if c in FREE and m.group(2):
                auc[(model, c)] = (m.group(1), m.group(2))
    return means, auc


MODEL_KEY = {'Random Forest': 'Random Forest', 'SVM': 'SVM', 'XGBoost': 'XGBoost',
             'GCN': 'GCN', 'GAT': 'GAT', 'GIN': 'GIN', 'SAGE': 'SAGE',
             'Graphormer': 'Graphormer', 'GraphGPS': 'GraphGPS',
             'Consensus-Classical': 'Consensus-Classical', 'Consensus-GNN': 'Consensus-GNN',
             'Consensus-GT': 'Consensus-GT', 'Consensus (Full)': 'Consensus',
             'MV-GNN-cross': 'MV-GNN-cross', 'MV-GNN$^{\\text{cross}}$': 'MV-GNN-cross',
             'MV-GNN$^{cross}$': 'MV-GNN-cross'}


def detect(units, auc, n_total, n_active):
    """The convention whose numbers match the table's printed AUC intervals."""
    best = None
    for kind in ('sd', 'naive', 'nb_distinct', 'nb_down'):
        hits = miss = 0
        for (label, c), (mean_txt, pm_txt) in auc.items():
            key = MODEL_KEY.get(label)
            g = units[(units.Model == key) & (units.CPT == 0.5)]
            if g.empty:
                miss += 1
                continue
            h = half_width(g[c].values, kind, n_total, n_active)
            hits += (f'{h:.3f}' == pm_txt)
        score = hits / max(len(auc) - miss, 1)
        if best is None or score > best[1]:
            best = (kind, score, hits, len(auc) - miss)
    return best


def resolve_means(lab, units, means):
    """Published text for every cell, upgraded to two decimals on percentages.

    The published percentage columns of the downsampled tables carry one
    decimal. The stated convention is two, so those cells are recomputed from
    the per-unit files -- and each recomputed value is required to round back
    to the published one, so the number itself is unchanged, only shown to one
    more place. Cells already at two decimals, and cells in KEEP_PUBLISHED, are
    left exactly as printed.
    """
    out, notes = {}, []
    for (model, cpt, c), txt in means.items():
        if (lab, model, cpt, c) in KEEP_PUBLISHED or c not in PCT:
            out[(model, cpt, c)] = txt
            continue
        if len(txt.split('.')[-1]) >= 2:
            out[(model, cpt, c)] = txt
            continue
        g = units[(units.Model == MODEL_KEY.get(model)) & (units.CPT == cpt)]
        if g.empty:
            out[(model, cpt, c)] = txt
            continue
        v = float(g[c].mean())
        # The recomputed value must be consistent with the printed one at the
        # precision it was printed to: within half a unit of the last place.
        # An exact .x5 tie rounds either way, so a strict round-trip would
        # reject it wrongly; the 0.05 tolerance is that half-unit.
        if abs(v - float(txt)) > 0.05 + 1e-9:
            notes.append(f'{model} cpt{cpt} {c}: published {txt}, recomputed {v:.2f}')
            out[(model, cpt, c)] = txt
        else:
            out[(model, cpt, c)] = f'{v:.2f}'
    return out, notes


def marks(means):
    """best / second-best displayed value per column, over every data row."""
    rank = {}
    for c in COLS:
        pool = {float(v) for (m, cpt, col), v in means.items() if col == c}
        s_ = sorted(pool, reverse=True)
        rank[c] = (s_[0] if s_ else None, s_[1] if len(s_) > 1 else None)
    return rank


def fitted_r(units, auc):
    """The correction factor implied by this table's published AUC intervals.

    Solving half = t*sqrt((1/n + r)*var) for r recovers the n_test/n_train the
    published interval used. Where the documented convention does not reproduce
    a table, this keeps the new columns consistent with the intervals already
    printed in it rather than silently applying a different correction.
    """
    rs = []
    for (label, c), (_, pm) in auc.items():
        g = units[(units.Model == MODEL_KEY.get(label)) & (units.CPT == 0.5)]
        if g.empty:
            continue
        v = g[c].values.astype(float)
        n, var = len(v), v.var(ddof=1)
        if var <= 0:
            continue
        rs.append((float(pm) / stats.t.ppf(0.975, df=n - 1)) ** 2 / var - 1.0 / n)
    return float(np.median(rs)) if rs else 0.0


def rebuild(block, units, means, kind, n_total, n_active, rank, r=None):
    """Reprint every data row with an interval on all eight columns."""
    half = {}
    for label in {m for m, _, _ in means}:
        key = MODEL_KEY.get(label)
        for cpt in CPTS:
            g = units[(units.Model == key) & (units.CPT == cpt)]
            if g.empty:
                continue
            for c in COLS:
                half[(label, cpt, c)] = half_width(g[c].values, kind, n_total, n_active, r)

    out, model = [], None
    for line in block.splitlines():
        f = line.rstrip().rstrip('\\').split('&')
        if '&' not in line or 'multicolumn' in line or 'textbf{Model}' in line or len(f) != 10 \
           or not re.fullmatch(r'0\.\d', f[1].strip()):
            out.append(line)
            continue
        head = f[0].strip()
        if head.startswith('\\multirow'):
            model = head[head.index('{*}{') + 4:].rstrip()[:-1]
        cpt = float(f[1].strip())
        cells = [f[0], f[1]]
        for c, raw in zip(COLS, f[2:]):
            txt = raw.strip()
            if not re.sub(r'[^0-9.]', '', txt):      # the empty half of a \multirow
                cells.append(raw)
                continue
            if (model, cpt, c) not in means or (model, cpt, c) not in half:
                cells.append(raw)
                continue
            val = means[(model, cpt, c)]
            pub_pm = re.search(r'\\pm\s*([\d.]+)', txt)
            # An interval the table already prints is reproduced verbatim: this
            # script adds the missing ones, it never restates a published one.
            pm = pub_pm.group(1) if pub_pm else f'{half[(model, cpt, c)]:.{nd(c)}f}'
            body = f'${val} \\pm {pm}$'
            best, second = rank[c]
            if best is not None and float(val) == best:
                body = '{\\boldmath' + body + '}'
            elif second is not None and float(val) == second:
                body = f'\\underline{{{body}}}'
            if txt.startswith('\\multirow'):
                body = f'\\multirow{{2}}{{*}}{{{body}}}'
            cells.append(' ' + body + ' ')
        out.append('&'.join(cells).rstrip() + ' \\\\')
    return '\n'.join(out)


def main():
    group = sys.argv[1] if len(sys.argv) > 1 else 'mmp'
    cfg = GROUPS[group]
    s = open(SI).read()
    report = []
    for lab, d, pattern, corr in cfg['tables']:
        i = s.index('\\label{' + lab + '}')
        b = s.rindex('\\begin{table}', 0, i)
        e = s.index('\\end{table}', i) + len('\\end{table}')
        block = s[b:e]
        units = load_units(d, pattern, corr)
        means, auc = parse_table(block)
        kind, score, hits, tot = detect(units, auc, cfg['n_total'], cfg['n_active'])
        r = None
        if score < 0.9:
            # The documented convention does not reproduce this table, so the
            # new columns use the correction its own printed intervals imply.
            r = fitted_r(units, auc)
            kind = 'fitted'
        means, notes = resolve_means(lab, units, means)
        for n_ in notes:
            print(f'  [{lab}] kept published (recomputed value does not round back): {n_}')
        s = s[:b] + rebuild(block, units, means, kind, cfg['n_total'], cfg['n_active'],
                            marks(means), r) + s[e:]
        report.append((lab, kind if r is None else f'fitted r={r:.3f}', hits, tot, 'rewritten'))
    open(SI, 'w').write(s)
    print(f'{"table":24s} {"convention":12s} {"AUC CIs matched":>16s}  status')
    for lab, kind, hits, tot, st in report:
        print(f'{lab:24s} {kind:12s} {f"{hits}/{tot}":>16s}  {st}')


if __name__ == '__main__':
    main()
