"""Collect per-unit ROC-AUC and PR-AUC for the HepG2 caspase assay across all
five split protocols and all eleven models, with Nadeau-Bengio corrected CIs.

Reconstructed after the original shared loader (gen5) was lost with its
scratchpad. Directory mapping was recovered by matching Random Forest ROC-AUC
means against the published figure, and is asserted in verify() below.
"""
import os, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

REB = '/Users/yamini/Desktop/projects/qsar casepase/qsar_prediction/Rebuttal'
WJ  = f'{REB}/summary_2_1437034/modeling/wahab_jobs'
UP  = f'{REB}/summary_2_1437034/upsampling'
COR = f'{REB}/graph_transformer_corrected/results/hepg2'

# Variance correction ratio n_test/n_train for the Nadeau-Bengio interval.
# It differs by protocol and must not be shared:
#   down-sampled (random_repeat / loo / scaffold): training is cut to a 1:1
#     set, 2 * 0.8 * n_active compounds, so the ratio is large. This branch
#     reproduces results/ci_summary.csv exactly.
#   up-sampled (up_single / up_loo): training keeps the full 8:1:1 split, so
#     n_test/n_train = 0.1/0.8 = 1/8, which is what the SI performance tables
#     use (acs-si-2.tex, tab:hepg2_up_loo).
N_TOTAL, N_ACTIVE = 7285, 282          # Active + Inactive, AID_654+1347034
DOWN_RATIO = (N_TOTAL / 10.0) / (2.0 * (0.8 * N_ACTIVE))
UP_RATIO = 1.0 / 8.0
UPSAMPLED = {'up_single', 'up_loo'}

PROTOS = [('random_repeat', 'Repeated random\n(5 repeats)'),
          ('loo',           '10-fold CV'),
          ('scaffold',      'Scaffold\n(5 repeats)'),
          ('up_single',     'Up-sampled\nsingle split'),
          ('up_loo',        'Up-sampled\n10-fold CV')]

# {proto: (directory, filename template)} for classical / GNN / MV-GNN
BASE = {
 'random_repeat': (f'{REB}/wahab_random_repeat/hepg2', '{k}_random_repeat_per_repeat.csv'),
 'loo':           (f'{WJ}/results_loo',                '{k}_loo_per_fold.csv'),
 'scaffold':      (f'{WJ}/results_scaffold',           '{k}_scaffold_per_repeat.csv'),
 'up_single':     (f'{UP}/results',                    '{k}_results.csv'),
 'up_loo':        (f'{UP}/results_loo',                '{k}_loo_per_fold.csv'),
}
KEYS = [('rf','Random Forest'),('svm','SVM'),('xgb','XGBoost'),
        ('gcn','GCN'),('gat','GAT'),('gin','GIN'),('sage','SAGE'),
        ('mvgnn','MV-GNN-cross')]
MODEL_ROW = {'rf':'Random Forest','svm':'SVM','xgb':'XGBoost','gcn':'GCN','gat':'GAT',
             'gin':'GIN','sage':'SAGE','mvgnn':'MV-GNN-cross'}
ORDER = ['Random Forest','SVM','XGBoost','GCN','GAT','GIN','SAGE',
         'Graphormer','GraphGPS','MV-GNN-cross','Consensus']
FAMILY = {'Random Forest':'Classical','SVM':'Classical','XGBoost':'Classical',
          'GCN':'GNN','GAT':'GNN','GIN':'GNN','SAGE':'GNN',
          'Graphormer':'GT','GraphGPS':'GT','MV-GNN-cross':'MV-GNN','Consensus':'Consensus'}
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
    """{proto: {model: {metric: np.array of per-unit values}}}"""
    out = {}
    for proto, _ in PROTOS:
        base, tmpl = BASE[proto]
        cols = {}
        for key, lab in KEYS:
            f = os.path.join(base, tmpl.format(k=key))
            if not os.path.exists(f):
                continue
            d = _test_rows(pd.read_csv(f), MODEL_ROW[key])
            if len(d):
                cols[lab] = {m: d[m].to_numpy(float) for m in METRICS}
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


def corrected_moe(v, proto):
    """Nadeau-Bengio corrected margin of error, as compute_ci_significance.py."""
    n = len(v)
    if n < 2:
        return 0.0
    var = float(np.var(v, ddof=1))
    ratio = UP_RATIO if proto in UPSAMPLED else DOWN_RATIO
    se = float(np.sqrt((1.0 / n + ratio) * var))
    return float(stats.t.ppf(0.975, df=n - 1) * se)


# ── bootstrap CI for the single-split protocol (one unit, no across-unit var) ──
PROB = {'Random Forest':'rf','SVM':'svm','XGBoost':'xgb','GCN':'gcn','GAT':'gat',
        'GIN':'gin','SAGE':'sage','Graphormer':'graphormer','GraphGPS':'graphgps',
        'MV-GNN-cross':'mvgnn','Consensus':'consensus'}

def bootstrap_single(n_boot=2000, seed=0):
    y = np.load(f'{UP}/data/y_test.npy')
    rng = np.random.default_rng(seed)
    idx = np.array([rng.choice(len(y), len(y), replace=True) for _ in range(n_boot)])
    out = {}
    for lab, key in PROB.items():
        f = f'{UP}/results/{key}_test_probs.npy'
        if not os.path.exists(f):
            continue
        p = np.load(f)
        if len(p) != len(y):
            continue
        roc, pr = [], []
        for b in idx:
            yb = y[b]
            if yb.min() == yb.max():
                continue
            roc.append(roc_auc_score(yb, p[b]))
            pr.append(average_precision_score(yb, p[b]))
        out[lab] = {'ROC-AUC': (np.percentile(roc, 2.5), np.percentile(roc, 97.5)),
                    'PR-AUC':  (np.percentile(pr, 2.5),  np.percentile(pr, 97.5))}
    return out


def table():
    """Tidy frame: proto, model, metric, mean, lo, hi, n, ci_kind."""
    pu, boot = per_unit(), bootstrap_single()
    rows = []
    for proto, _ in PROTOS:
        for model in ORDER:
            if model not in pu[proto]:
                continue
            for m in METRICS:
                v = pu[proto][model][m]
                mu, n = float(np.mean(v)), len(v)
                if n > 1:
                    half = corrected_moe(v, proto)
                    lo, hi, kind = mu - half, mu + half, 'corrected-t'
                elif model in boot:
                    lo, hi = boot[model][m]; kind = 'bootstrap'
                else:
                    lo = hi = mu; kind = 'point'
                rows.append(dict(proto=proto, model=model, metric=m, mean=mu,
                                 lo=lo, hi=hi, n=n, ci=kind))
    return pd.DataFrame(rows)


PUBLISHED_ROC = {  # read off the existing figures/hepg2_main_results.png
 'random_repeat': {'Random Forest':.862,'SVM':.870,'XGBoost':.832,'GCN':.837,'GAT':.824,
                   'GIN':.848,'SAGE':.839,'Graphormer':.831,'GraphGPS':.827,'Consensus':.884},
 'loo':  {'Random Forest':.845,'SVM':.850,'XGBoost':.836,'GCN':.828,'GAT':.829,'GIN':.837,
          'SAGE':.828,'Graphormer':.829,'GraphGPS':.841,'MV-GNN-cross':.831,'Consensus':.876},
 'scaffold': {'Random Forest':.761,'SVM':.790,'XGBoost':.741,'GCN':.754,'GAT':.750,'GIN':.742,
              'SAGE':.753,'Graphormer':.794,'GraphGPS':.817,'MV-GNN-cross':.707,'Consensus':.812},
 'up_single':{'Random Forest':.897,'SVM':.881,'XGBoost':.879,'GCN':.806,'GAT':.814,'GIN':.854,
              'SAGE':.815,'Graphormer':.791,'GraphGPS':.853,'MV-GNN-cross':.906,'Consensus':.865},
 'up_loo':  {'Random Forest':.882,'SVM':.853,'XGBoost':.863,'GCN':.864,'GAT':.865,'GIN':.864,
             'SAGE':.870,'Graphormer':.846,'GraphGPS':.865,'MV-GNN-cross':.871,'Consensus':.897},
}

def verify(t):
    bad = []
    for proto, exp in PUBLISHED_ROC.items():
        for model, want in exp.items():
            r = t[(t.proto == proto) & (t.model == model) & (t.metric == 'ROC-AUC')]
            if not len(r):
                bad.append((proto, model, want, None)); continue
            got = round(float(r['mean'].iloc[0]), 3)
            if abs(got - want) > 0.0011:
                bad.append((proto, model, want, got))
    return bad
