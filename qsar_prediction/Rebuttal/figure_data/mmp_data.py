"""Per-unit ROC-AUC and PR-AUC for the MMP summary assay (AID 720637),
with Nadeau-Bengio corrected CIs. Same interface as hepg2_data.

The up-sampled protocol runs under Summary_1_models/upsampling_protocols used
gt_model_corrected.py, so Graphormer and GraphGPS there are already the
corrected architectures and no merge with graph_transformer_corrected/ is
needed. The down-sampled protocols take their transformers and consensus from
graph_transformer_corrected/results/mmp/, as the published figure does.
"""
import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

REB = '/Users/yamini/Desktop/projects/qsar casepase/qsar_prediction/Rebuttal'
S1  = f'{REB}/Summary_1_models'
WJ  = f'{S1}/modeling/wahab_jobs'
UPP = f'{S1}/upsampling_protocols'
COR = f'{REB}/graph_transformer_corrected/results/mmp'

# Variance correction ratio n_test/n_train for the Nadeau-Bengio interval.
# The up-sampled protocol trains on an 8:1:1 train/val/test split inside each
# fold, so the held-out tenth is measured against eight tenths of training
# data: n_test/n_train = 0.1/0.8 = 1/8. This is the ratio the SI performance
# tables use (acs-si-2.tex, tab:mmp_up_loo), and verify() checks every value
# against that table. It is NOT 1/9: the validation tenth is not trained on.
TEST_TRAIN = 1.0 / 8.0

PROTOS = [('up_loo', 'Up-sampled 10-fold CV')]

# classical / GNN / MV-GNN (and, for up_* protocols, GT + consensus too)
BASE = {
 'up_loo':           (f'{UPP}/results_loo',            '{k}_per_unit.csv'),
 'up_random_repeat': (f'{UPP}/results_random_repeat',  '{k}_per_unit.csv'),
 'up_scaffold':      (f'{UPP}/results_scaffold',       '{k}_per_unit.csv'),
 'up_single':        (f'{S1}/upsampling/results',      '{k}_results.csv'),
 'loo':              (f'{WJ}/results_loo',             '{k}_loo_per_fold.csv'),
 'scaffold':         (f'{WJ}/results_scaffold',        '{k}_scaffold_per_repeat.csv'),
}
SELF_CONTAINED = {'up_loo', 'up_random_repeat', 'up_scaffold', 'up_single'}

KEYS = [('rf','Random Forest'),('svm','SVM'),('xgb','XGBoost'),
        ('gcn','GCN'),('gat','GAT'),('gin','GIN'),('sage','SAGE'),
        ('graphormer','Graphormer'),('graphgps','GraphGPS'),
        ('mvgnn','MV-GNN-cross'),('consensus','Consensus')]
ORDER = ['Random Forest','SVM','XGBoost','GCN','GAT','GIN','SAGE',
         'Graphormer','GraphGPS','MV-GNN-cross','Consensus']
FAMILY = {'Random Forest':'Classical','SVM':'Classical','XGBoost':'Classical',
          'GCN':'GNN','GAT':'GNN','GIN':'GNN','SAGE':'GNN',
          'Graphormer':'GT','GraphGPS':'GT','MV-GNN-cross':'MV-GNN',
          'Consensus':'Consensus'}
CORRECTED = {'Graphormer','GraphGPS','Consensus'}
METRICS = ['ROC-AUC','PR-AUC']


def _test_rows(df, model=None):
    d = df[df['Split'] == 'Test']
    if 'CPT' in d.columns:
        d = d[d['CPT'] == 0.5]
    if model is not None and 'Model' in d.columns:
        d = d[d['Model'] == model]
    return d


def per_unit():
    out = {}
    for proto, _ in PROTOS:
        base, tmpl = BASE[proto]
        cols = {}
        keys = KEYS if proto in SELF_CONTAINED else KEYS[:7] + [('mvgnn', 'MV-GNN-cross')]
        for key, lab in keys:
            f = os.path.join(base, tmpl.format(k=key))
            if not os.path.exists(f):
                continue
            d = _test_rows(pd.read_csv(f), lab)
            if len(d):
                unit = next((c for c in ('Unit', 'Fold', 'Repeat') if c in d.columns), None)
                if unit:
                    d = d.sort_values(unit)
                cols[lab] = {m: d[m].to_numpy(float) for m in METRICS}
        if proto not in SELF_CONTAINED:
            pu = f'{COR}/{proto}/per_unit_all.csv'
            if os.path.exists(pu):
                d = pd.read_csv(pu)
                for lab in ('Graphormer', 'GraphGPS'):
                    g = _test_rows(d, lab).sort_values('Unit')
                    if len(g):
                        cols[lab] = {m: g[m].to_numpy(float) for m in METRICS}
            cm = f'{COR}/{proto}/consensus_metrics.csv'
            if os.path.exists(cm):
                g = _test_rows(pd.read_csv(cm), 'Consensus').sort_values('Unit')
                if len(g):
                    cols['Consensus'] = {m: g[m].to_numpy(float) for m in METRICS}
        out[proto] = cols
    return out


def corrected_moe(v):
    n = len(v)
    if n < 2:
        return 0.0
    var = float(np.var(v, ddof=1))
    se = float(np.sqrt((1.0 / n + TEST_TRAIN) * var))
    return float(stats.t.ppf(0.975, df=n - 1) * se)


def table():
    pu = per_unit()
    rows = []
    for proto, _ in PROTOS:
        for model in ORDER:
            if model not in pu[proto]:
                continue
            for m in METRICS:
                v = pu[proto][model][m]
                mu, n = float(np.mean(v)), len(v)
                half = corrected_moe(v)
                rows.append(dict(proto=proto, model=model, metric=m, mean=mu,
                                 lo=mu - half, hi=mu + half, n=n,
                                 ci='corrected-t' if n > 1 else 'point'))
    return pd.DataFrame(rows)


# acs-si-2.tex, Table "(Part 4 of 6)" tab:mmp_up_loo -- mean and 95% CI
# half-width, as printed. The figure must agree with the SI to 3 decimals.
SI_UP_LOO = {
 'Random Forest': ((0.897, 0.026), (0.678, 0.051)),
 'SVM':           ((0.903, 0.023), (0.701, 0.046)),
 'XGBoost':       ((0.891, 0.024), (0.674, 0.060)),
 'GCN':           ((0.915, 0.018), (0.691, 0.072)),
 'GAT':           ((0.906, 0.017), (0.681, 0.077)),
 'GIN':           ((0.923, 0.017), (0.728, 0.058)),
 'SAGE':          ((0.912, 0.019), (0.700, 0.072)),
 'Graphormer':    ((0.880, 0.027), (0.607, 0.075)),
 'GraphGPS':      ((0.914, 0.017), (0.707, 0.043)),
 'MV-GNN-cross':  ((0.907, 0.026), (0.697, 0.062)),
 'Consensus':     ((0.932, 0.016), (0.772, 0.051)),
}


def verify(t):
    """Every plotted mean and interval must reproduce the SI table exactly."""
    bad = []
    for model, (roc, pr) in SI_UP_LOO.items():
        for metric, (wmean, wmoe) in (('ROC-AUC', roc), ('PR-AUC', pr)):
            r = t[(t.proto == 'up_loo') & (t.model == model) & (t.metric == metric)]
            if not len(r):
                bad.append((model, metric, 'missing')); continue
            gmean = round(float(r['mean'].iloc[0]), 3)
            gmoe = round(float((r['hi'].iloc[0] - r['lo'].iloc[0]) / 2), 3)
            if (gmean, gmoe) != (wmean, wmoe):
                bad.append((model, metric, (wmean, wmoe), (gmean, gmoe)))
    return bad
