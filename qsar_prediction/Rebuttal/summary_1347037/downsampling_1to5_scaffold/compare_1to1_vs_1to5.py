"""Compares the 1:5 downsampled CHO-K1 scaffold run with the published 1:1 run.

Both use the SAME 5 scaffold test sets, so per-repeat differences are paired.
1:1 rows: classical / GNN / MV-GNN from ../modeling/wahab_jobs/results_scaffold;
Graphormer, GraphGPS and every consensus group from the CORRECTED retrain in
../../graph_transformer_corrected/results/chok1/scaffold (so both sides use the
corrected graph transformers).

Usage: python compare_1to1_vs_1to5.py
Output: results_scaffold/compare_1to1_vs_1to5.csv
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, '..', 'modeling', 'wahab_jobs', 'results_scaffold')
GTC = os.path.join(HERE, '..', '..', 'graph_transformer_corrected', 'results', 'chok1', 'scaffold')
NEW = os.path.join(HERE, 'results_scaffold')
METRICS = ['ROC-AUC', 'PR-AUC', 'MCC', 'CCR']
CPT = 0.5

MODELS = [('rf', 'Random Forest'), ('svm', 'SVM'), ('xgb', 'XGBoost'), ('gcn', 'GCN'),
          ('gat', 'GAT'), ('gin', 'GIN'), ('sage', 'SAGE'), ('mvgnn', 'MV-GNN-cross'),
          ('graphormer', 'Graphormer'), ('graphgps', 'GraphGPS')]
CONS = ['Consensus-Classical', 'Consensus-GNN', 'Consensus-GT', 'Consensus', 'Consensus-MVGNN']


def test_rows(df, unit_col):
    d = df[(df['Split'] == 'Test') & (df['CPT'] == CPT)] if 'Split' in df else df[df['CPT'] == CPT]
    return d.set_index(unit_col)[METRICS].sort_index()


def old_per_repeat(key, name):
    if key in ('graphormer', 'graphgps'):
        rows = []
        for r in range(5):
            m = pd.read_csv(os.path.join(GTC, f'{name}_{r}_metrics.csv'))
            m = m[(m['Split'] == 'Test') & (m['CPT'] == CPT)].iloc[0]
            rows.append({'Repeat': r, **{k: m[k] for k in METRICS}})
        return pd.DataFrame(rows).set_index('Repeat')[METRICS]
    return test_rows(pd.read_csv(os.path.join(OLD, f'{key}_scaffold_per_repeat.csv')), 'Repeat')


def old_consensus():
    p = os.path.join(GTC, 'per_unit_all.csv')
    d = pd.read_csv(p)
    unit = 'Unit' if 'Unit' in d else 'Repeat'
    return {n: test_rows(d[d['Model'] == n], unit) for n in CONS}


rows = []
new_cons = pd.read_csv(os.path.join(NEW, 'consensus_per_unit.csv'))
oc = old_consensus()
pairs = [(name, old_per_repeat(k, name), test_rows(pd.read_csv(os.path.join(NEW, f'{k}_per_unit.csv')), 'Repeat'))
         for k, name in MODELS]
pairs += [(n, oc[n], test_rows(new_cons[new_cons['Model'] == n], 'Repeat')) for n in CONS]
for name, o, n in pairs:
    row = {'Model': name}
    for m in METRICS:
        diff = n[m].values - o[m].values
        row[f'{m} 1:1'] = f'{o[m].mean():.3f} ± {o[m].std():.3f}'
        row[f'{m} 1:5'] = f'{n[m].mean():.3f} ± {n[m].std():.3f}'
        row[f'{m} diff'] = round(diff.mean(), 4)
        row[f'{m} 1:5 better (of 5)'] = int((diff > 0).sum())
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(NEW, 'compare_1to1_vs_1to5.csv'), index=False)
pd.set_option('display.width', 250)
print(out[['Model', 'ROC-AUC 1:1', 'ROC-AUC 1:5', 'ROC-AUC diff', 'ROC-AUC 1:5 better (of 5)',
           'PR-AUC diff', 'MCC diff', 'CCR diff']].to_string(index=False))
