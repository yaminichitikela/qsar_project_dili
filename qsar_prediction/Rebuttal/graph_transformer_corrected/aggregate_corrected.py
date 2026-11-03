"""Pools the per-unit metrics into summaries, in the published file format.

Emits the same shape of file the published pipeline produces
(Model, CPT, Stat=mean/std, then the metric columns), so the downstream chain --
compute_ci_significance.py, build_master_workbook.py, build_rebuttal_latex_tables.py
-- can read these in place of the old ones without changes.

Also writes a side-by-side old/new comparison per assay and protocol, which is
the table to look at first: it shows what correcting the architecture actually
did to Graphormer, GraphGPS, Consensus-GT, Consensus and Consensus-MVGNN, with
the other seven families unchanged by construction.

Usage:
    python aggregate_corrected.py                      # every assay/protocol found
    python aggregate_corrected.py mmp                  # one assay
    python aggregate_corrected.py mmp loo              # one assay + protocol
Outputs (./results/<assay>/<protocol>/):
    <arch>_summary.csv, consensus_summary.csv
    per_unit_all.csv
Outputs (./results/):
    old_vs_new_all.csv      -- every assay x protocol x changed model
"""
import glob
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_map import ASSAY_LABEL, ASSAY_PROTOCOLS, PROTOCOLS, out_dir

METRICS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC',
           'Bal-Acc', 'F1']
CHANGED = ['Graphormer', 'GraphGPS', 'Consensus-GT', 'Consensus', 'Consensus-MVGNN']


def summarise(df, label):
    """mean/std across units, per (Model, CPT) -- the published summary layout."""
    out = []
    for (model, cpt), g in df.groupby(['Model', 'CPT'], sort=False):
        for stat, fn in (('mean', np.mean), ('std', lambda x: np.std(x, ddof=1))):
            row = {'Model': model, 'CPT': cpt, 'Stat': stat}
            for m in METRICS:
                v = fn(g[m].values) if len(g) > 1 or stat == 'mean' else 0.0
                row[m] = round(float(v), 4 if m in ('ROC-AUC', 'PR-AUC', 'MCC', 'F1') else 2)
            out.append(row)
    res = pd.DataFrame(out)
    print(f'    {label}: {df.Model.nunique()} models x {df.Unit.nunique()} units')
    return res


def do(assay, protocol, old_new_rows):
    d = out_dir(assay, protocol)
    if not os.path.isdir(d):
        return False
    per_unit = []
    for f in sorted(glob.glob(f'{d}/*_metrics.csv')):
        if os.path.basename(f).startswith('consensus'):
            continue
        per_unit.append(pd.read_csv(f))
    cons = f'{d}/consensus_metrics.csv'
    if os.path.exists(cons):
        per_unit.append(pd.read_csv(cons))
    if not per_unit:
        return False

    df = pd.concat(per_unit, ignore_index=True)
    test = df[df.Split == 'Test'].copy()
    print(f'  {assay}/{protocol}')
    test.to_csv(f'{d}/per_unit_all.csv', index=False)

    for arch in ('Graphormer', 'GraphGPS'):
        sub = test[test.Model == arch]
        if len(sub):
            summarise(sub, arch).to_csv(f'{d}/{arch}_summary.csv', index=False)
    consub = test[test.Model.str.startswith('Consensus')]
    if len(consub):
        summarise(consub, 'consensus').to_csv(f'{d}/consensus_summary.csv', index=False)

    # old vs new for the rows the correction touches
    cmp_path = f'{d}/gt_old_vs_new_{protocol}.csv'
    if os.path.exists(cmp_path):
        c = pd.read_csv(cmp_path)
        g = c.groupby('model')[['old_auc', 'new_auc']].agg(['mean', 'std'])
        for model in c.model.unique():
            old_m = g.loc[model, ('old_auc', 'mean')]
            new_m = g.loc[model, ('new_auc', 'mean')]
            old_networks = c[c.model == model]
            old_new_rows.append(dict(
                assay=assay, protocol=protocol, model=model,
                n_units=len(old_networks),
                old_roc_auc=round(float(old_m), 4),
                new_roc_auc=round(float(new_m), 4),
                delta=round(float(new_m - old_m), 4),
                new_std=round(float(g.loc[model, ('new_auc', 'std')] or 0), 4)))
    return True


if __name__ == '__main__':
    args = sys.argv[1:]
    assays = [a for a in args if a in ASSAY_PROTOCOLS] or list(ASSAY_PROTOCOLS)
    protos = [p for p in args if p in PROTOCOLS] or list(PROTOCOLS)

    rows, found = [], 0
    for assay in assays:
        for protocol in [p for p in ASSAY_PROTOCOLS[assay] if p in protos]:
            if do(assay, protocol, rows):
                found += 1
    if not found:
        print('No per-unit metrics found -- run train_gt_corrected.py and '
              'compute_consensus_corrected.py first')
        sys.exit(1)

    if rows:
        cmp_df = pd.DataFrame(rows)
        os.makedirs(f'{HERE}/results', exist_ok=True)
        cmp_df.to_csv(f'{HERE}/results/old_vs_new_all.csv', index=False)
        print('\n' + '=' * 78)
        print('EFFECT OF CORRECTING THE ARCHITECTURE (test ROC-AUC)')
        print('=' * 78)
        piv = cmp_df.pivot_table(index=['assay', 'protocol'], columns='model',
                                 values='delta')
        cols = [c for c in CHANGED if c in piv.columns]
        print(piv[cols].round(4).to_string())
        print('\nmean delta per model, across everything aggregated so far:')
        print(cmp_df.groupby('model')['delta'].agg(['mean', 'min', 'max'])
              .round(4).to_string())
        print(f'\n-> results/old_vs_new_all.csv')
