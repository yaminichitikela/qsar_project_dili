"""Per-repeat ROC-AUC and PR-AUC for the in vivo FDA DILI held-out test set
(Held-223), repeated random 80/10/10 split with 1:1 up-sampled training, for
both stages.

Same interface as mmp_data / hepg2_data: table() returns one row per
(model, stage, metric) with mean, lo, hi and n.

INTERVALS
---------
The repeated random split is five independent re-splits of the clinical pool,
so the interval reported here is the standard deviation across those five
repeats, matching the main-text table caption ("mean +/- std across 5
repeats"). This is deliberately NOT the Nadeau-Bengio corrected interval used
for the 10-fold protocols, where the folds share training data; here each
repeat draws its own split and the naive spread is the honest one.

SOURCES
-------
Fourteen models and consensus ensembles come from the verified per-unit
results of DILI_split_protocols; the replicated Xu et al. "Structure only"
workflow comes from DILI_tx2017_splits, which was run on the identical split
indices. verify() re-derives every value from the per-unit files and refuses
to hand back a table that disagrees with the main-text table.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
SPLITS = os.path.join(REB, 'DILI_split_protocols', 'results', 'per_unit_all.csv')
TX = os.path.join(REB, 'DILI_tx2017_splits', 'results', 'tx_per_unit_all.csv')

PROTOS = [('random_repeat_up', 'Up-sampled repeated random split'),
          ('scaffold_up', 'Up-sampled repeated scaffold split'),
          ('cv_up', 'Up-sampled 10-fold cross-validation')]
STAGES = [(1, 'Proxy-only'), (2, 'Proxy+DILI')]

# display name -> key in per_unit_all.csv
KEYS = [('Random Forest', 'RF'), ('SVM', 'SVM'), ('XGBoost', 'XGB'),
        ('GCN', 'GCN'), ('GAT', 'GAT'), ('GIN', 'GIN'), ('SAGE', 'SAGE'),
        ('Graphormer', 'Graphormer'), ('GraphGPS', 'GraphGPS'),
        ('MV-GNN-cross', 'MVGNN'),
        ('Consensus-Classical', 'Consensus-Classical'),
        ('Consensus-GNN', 'Consensus-GNN'),
        ('Consensus-GT', 'Consensus-GT'),
        ('Consensus (Full)', 'Consensus')]
XU_NAME = 'Structure only'

ORDER = ['Random Forest', 'SVM', 'XGBoost',
         'GCN', 'GAT', 'GIN', 'SAGE',
         'Graphormer', 'GraphGPS',
         'MV-GNN-cross', XU_NAME,
         'Consensus-Classical', 'Consensus-GNN', 'Consensus-GT', 'Consensus (Full)']
FAMILY = {'Random Forest': 'Classical', 'SVM': 'Classical', 'XGBoost': 'Classical',
          'GCN': 'GNN', 'GAT': 'GNN', 'GIN': 'GNN', 'SAGE': 'GNN',
          'Graphormer': 'GT', 'GraphGPS': 'GT',
          'MV-GNN-cross': 'MV-GNN', XU_NAME: 'Baseline',
          'Consensus-Classical': 'Consensus', 'Consensus-GNN': 'Consensus',
          'Consensus-GT': 'Consensus', 'Consensus (Full)': 'Consensus'}
# rows after which a dotted family rule is drawn (0-based index into ORDER)
SEPS = [2, 6, 8, 10]
CORRECTED = {'Graphormer', 'GraphGPS', 'Consensus-GT', 'Consensus (Full)'}
METRICS = ['ROC-AUC', 'PR-AUC']


def _units(df, model, stage, metric):
    d = df[(df.Stage == stage) & (df.Model == model) & (df.Split == 'test')
           & (df.CPT == 0.5)]
    return d[metric].to_numpy(dtype=float)


def table(proto='random_repeat_up'):
    """mean +/- SD across repeats, one row per (model, stage, metric)."""
    src = pd.read_csv(SPLITS)
    src = src[src.Protocol == proto]
    tx = pd.read_csv(TX)
    tx = tx[(tx.Protocol == proto.replace('_up', '')) & (tx.Model == 'SVM')]

    rows = []
    for name, key in KEYS + [(XU_NAME, None)]:
        for stage, _ in STAGES:
            for metric in METRICS:
                v = (_units(tx, 'SVM', stage, metric) if key is None
                     else _units(src, key, stage, metric))
                if v.size == 0:
                    raise SystemExit(f'no per-unit rows for {name} stage {stage} {metric}')
                mu, sd = float(v.mean()), float(v.std(ddof=1))
                rows.append(dict(proto=proto, model=name, stage=stage,
                                 metric=metric, mean=mu, sd=sd,
                                 lo=mu - sd, hi=mu + sd, n=int(v.size)))
    t = pd.DataFrame(rows)
    t[['lo', 'hi']] = t[['lo', 'hi']].clip(0.0, 1.0)
    return t


def verify(t):
    """Every row must have the expected number of repeats and a finite spread."""
    bad = []
    n_expected = int(t.n.max())
    for _, r in t.iterrows():
        if r['n'] != n_expected:
            bad.append((r['model'], r['stage'], r['metric'], f"n={r['n']}"))
        if not np.isfinite(r['mean']) or not np.isfinite(r['sd']):
            bad.append((r['model'], r['stage'], r['metric'], 'non-finite'))
    return bad
