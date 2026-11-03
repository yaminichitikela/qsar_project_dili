"""95% intervals on ALL EIGHT metrics for every DILI split-protocol model.

The tables in results/tables_dili_split_protocols_assay_style*.tex print the
+/- interval only on ROC-AUC and PR-AUC; every other column is a bare mean.
Nothing is missing from the raw results to do better -- results/per_unit_all.csv
carries all eight metrics for every model, protocol, stage and unit -- so this
script applies the same interval convention to the remaining six and writes the
numbers out for the rebuttal.

Interval convention is identical to make_assay_style_tables.interval:
    random_repeat*   standard deviation across the 5 repeats
    cv*, scaffold*   Nadeau--Bengio corrected 95% half-width,
                     t(0.975, n-1) * sqrt((1/n + n_test/n_train) * var)
with n_test the 222 Held-223 compounds and n_train the stage's mean
training-set size. Threshold metrics (Sensitivity, Specificity, CCR, MCC,
Bal-Acc, F1) depend on the cutoff, so they are reported per cutoff; ROC-AUC and
PR-AUC are threshold-free and identical at both cutoffs.

The percentage metrics are kept on the 0-100 scale the results use, so their
half-widths are in percentage points.

Usage: python compute_all_metric_ci.py
Output: results/ci_all_metrics_dili.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import data_dir, unit_dir
from make_assay_style_tables import (COLS, GROUPS, PCT, PROTO_SETS, XU_KEY,
                                     interval, load_xu, n_train, nd)

THRESH_FREE = ['ROC-AUC', 'PR-AUC']


def main():
    per_all = pd.read_csv(os.path.join(HERE, 'results', 'per_unit_all.csv'))
    ddir = data_dir()
    n_ncv = len(np.load(f'{ddir}/y_ncv.npy'))
    n_test = len(np.load(f'{ddir}/y_test.npy'))

    rows = []
    for which, protos in PROTO_SETS.items():
        for proto, n_units, desc, kind, _, _ in protos:
            for cpt in (0.5, 0.8):
                per = per_all[(per_all.Split == 'test')
                              & ((per_all.CPT - cpt).abs() < 1e-9)]
                xu = load_xu(proto, cpt)
                for stage in (1, 2):
                    sub = per[(per.Protocol == proto) & (per.Stage == stage)]
                    if xu is not None:
                        sub = pd.concat([sub, xu[xu.Stage == stage]], ignore_index=True)
                    ntr = n_train(proto, n_units, stage, n_ncv)
                    keys = [k for _, members in GROUPS for k, _ in members]
                    if xu is not None:
                        keys.append(XU_KEY)
                    for key in keys:
                        g = sub[sub.Model == key]
                        if g.empty:
                            continue
                        for c in COLS:
                            # threshold-free metrics are cutoff-invariant: report once
                            if c in THRESH_FREE and cpt != 0.5:
                                continue
                            vals = g[c].values.astype(float)
                            mean = float(vals.mean())
                            half = float(interval(vals, kind, ntr, n_test))
                            # The published tables round the mean to the column's
                            # precision BEFORE formatting it and leave the
                            # half-width raw; reproduce that so the Reported
                            # strings match the tables digit for digit.
                            shown = round(mean, nd(c))
                            rows.append({
                                'Balancing': which, 'Protocol': proto, 'Stage': stage,
                                'Model': key, 'Metric': c,
                                'CPT': None if c in THRESH_FREE else cpt,
                                'Units': len(vals), 'IntervalType': 'SD' if kind == 'sd' else 'NB95',
                                'Mean': round(mean, 4), 'SD': round(float(vals.std(ddof=1)), 4),
                                'Half_width': round(half, 4),
                                'CI95_Low': round(mean - half, 4),
                                'CI95_High': round(mean + half, 4),
                                'Reported': f'{shown:.{nd(c)}f} +/- {half:.{nd(c)}f}'
                                            + (' %' if c in PCT else ''),
                            })
    out = pd.DataFrame(rows)
    path = os.path.join(HERE, 'results', 'ci_all_metrics_dili.csv')
    out.to_csv(path, index=False)
    print(f'{len(out)} rows -> {path}')
    print(f"metrics: {sorted(out.Metric.unique())}")
    print(f"models:  {sorted(out.Model.unique())}")
    print(f"\nCovered: {out.groupby(['Balancing', 'Protocol']).Model.nunique().to_string()}")


if __name__ == '__main__':
    main()
