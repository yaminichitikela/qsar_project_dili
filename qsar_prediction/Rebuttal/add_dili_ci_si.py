"""Adds the five missing intervals to the DILI tables in the SI.

The DILI tables (S5, Parts 1-3) carry intervals on ROC-AUC and PR-AUC only,
while the assay tables now carry them on every metric. This closes that gap
under the same two rules used there:

  * the published mean is kept, and a percentage printed to one decimal is
    re-shown to two only when the recomputed value rounds back to it;
  * a published interval is reproduced verbatim -- the correction factor for
    the new columns is the one that table's own ROC-AUC and PR-AUC intervals
    imply, so the new columns are consistent with the printed ones.

Structure-only rows come from DILI_tx2017_splits/results_down (the SVM the
TX2017 search selects), the rest from DILI_split_protocols; both were verified
to reproduce all 630 published cells before anything was written.

Usage: python add_dili_ci_si.py
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

COLS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'F1']
PCT = {'Sensitivity', 'Specificity', 'CCR'}
FREE = {'ROC-AUC', 'PR-AUC'}
NAME = {'Random Forest': 'RF', 'SVM': 'SVM', 'XGBoost': 'XGB', 'GCN': 'GCN', 'GAT': 'GAT',
        'GIN': 'GIN', 'SAGE': 'SAGE', 'Graphormer': 'Graphormer', 'GraphGPS': 'GraphGPS',
        'MV-GNN': 'MVGNN', 'Consensus-Classical': 'Consensus-Classical',
        'Consensus-GNN': 'Consensus-GNN', 'Consensus-GT': 'Consensus-GT',
        'Consensus (Full)': 'Consensus', 'Structure only': 'XU'}
PROTO = {'tab:dili_random_assay_style': 'random_repeat',
         'tab:dili_cv_assay_style': 'cv',
         'tab:dili_scaffold_assay_style': 'scaffold'}


def nd(c):
    return 2 if c in PCT else 3


def sources():
    per = pd.read_csv(os.path.join(HERE, 'DILI_split_protocols', 'results', 'per_unit_all.csv'))
    per = per[(per.Split == 'test') & ((per.CPT - 0.5).abs() < 1e-9)]
    xu = pd.read_csv(os.path.join(HERE, 'DILI_tx2017_splits', 'results_down', 'tx_per_unit_all.csv'))
    xu = xu[(xu.Split == 'test') & ((xu.CPT - 0.5).abs() < 1e-9) & (xu.Model == 'SVM')].copy()
    xu['Model'] = 'XU'
    return per, xu


def rows_of(block):
    """(stage, printed name, fields) for every data row, in order."""
    out, stage = [], None
    for line in block.splitlines():
        if 'Stage 1' in line:
            stage = 1
        elif 'Stage 2' in line:
            stage = 2
        f = line.rstrip().rstrip('\\').split('&')
        if len(f) != len(COLS) + 1 or 'multicolumn' in line or 'textbf{Model}' in line:
            out.append((None, None, line))
            continue
        name = re.sub(r'\$\^.*|\\textbf\{|[{}$]', '', f[0]).strip()
        out.append((stage, name, f) if name in NAME else (None, None, line))
    return out


def main():
    per, xu = sources()
    s = open(SI).read()
    for lab, proto in PROTO.items():
        i = s.index('\\label{' + lab + '}')
        b = s.rindex('\\begin{table}', 0, i)
        e = s.index('\\end{table}', i)
        block = s[b:e]
        parsed = rows_of(block)

        series, pub = {}, {}
        for stage, name, f in parsed:
            if stage is None:
                continue
            key = NAME[name]
            src = xu if key == 'XU' else per
            g = src[(src.Protocol == proto) & (src.Stage == stage) & (src.Model == key)]
            assert not g.empty, f'{lab} s{stage} {name}'
            for c, raw in zip(COLS, f[1:]):
                series[(stage, name, c)] = g[c].values.astype(float)
                m = re.search(r'\\pm\s*([\d.]+)', raw)
                if m:
                    pub[(stage, name, c)] = m.group(1)

        # correction factor implied by the intervals this table already prints
        rs = []
        for k, pm in pub.items():
            v = series[k]
            n, var = len(v), v.var(ddof=1)
            if var <= 0:
                continue
            rs.append((float(pm) / stats.t.ppf(0.975, df=n - 1)) ** 2 / var - 1.0 / n)
        r = float(np.median(rs))

        def half(k):
            v = series[k]
            n = len(v)
            return stats.t.ppf(0.975, df=n - 1) * np.sqrt((1.0 / n + r) * v.var(ddof=1))

        # means, with percentages shown to two decimals where that rounds back
        val = {}
        for stage, name, f in parsed:
            if stage is None:
                continue
            for c, raw in zip(COLS, f[1:]):
                t = re.sub(r'\\textbf\{|\{\\boldmath|\\underline\{|\\boldmath|[{}$]', '', raw).strip()
                txt = re.match(r'([\d.]+)', t).group(1)
                k = (stage, name, c)
                if c in PCT and len(txt.split('.')[-1]) < 2:
                    v = float(series[k].mean())
                    txt = f'{v:.2f}' if abs(v - float(txt)) <= 0.05 + 1e-9 else txt
                val[k] = txt

        # best / second best per column WITHIN each stage, as the caption states
        rank = {}
        for stage in (1, 2):
            for c in COLS:
                pool = sorted({float(v) for (st, nm, col), v in val.items()
                               if st == stage and col == c}, reverse=True)
                rank[(stage, c)] = (pool[0] if pool else None,
                                    pool[1] if len(pool) > 1 else None)

        out = []
        for stage, name, f in parsed:
            if stage is None:
                out.append(f)
                continue
            cells = [f[0]]
            for c, raw in zip(COLS, f[1:]):
                k = (stage, name, c)
                pm = pub.get(k) or f'{half(k):.{nd(c)}f}'
                body = f'${val[k]} \\pm {pm}$'
                best, second = rank[(stage, c)]
                if best is not None and float(val[k]) == best:
                    body = '{\\boldmath' + body + '}'
                elif second is not None and float(val[k]) == second:
                    body = f'\\underline{{{body}}}'
                cells.append(' ' + body + ' ')
            out.append('&'.join(cells).rstrip() + ' \\\\')
        s = s[:b] + '\n'.join(out) + s[e:]
        print(f'{lab:32s} fitted r={r:.4f}  rewritten')
    open(SI, 'w').write(s)


if __name__ == '__main__':
    main()
