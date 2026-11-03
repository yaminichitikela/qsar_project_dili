"""Builds results/upsampling_1to5_all_results.txt: every up-sampled run on the
three summary assays plus the CHO-K1 1:5 scaffold run, all metrics, mean +/- std
across split units, with the best value in each column marked.

Usage: python build_upsampling_1to5_report.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, f1_score, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'upsampling_1to5_all_results.txt')
METRICS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
PCT = {'Sensitivity', 'Specificity', 'CCR', 'Bal-Acc'}
MODELS = [('rf', 'Random Forest'), ('svm', 'SVM'), ('xgb', 'XGBoost'), ('gcn', 'GCN'),
          ('gat', 'GAT'), ('gin', 'GIN'), ('sage', 'SAGE'), ('mvgnn', 'MV-GNN-cross'),
          ('graphormer', 'Graphormer'), ('graphgps', 'GraphGPS')]
CONS = ['Consensus-Classical', 'Consensus-GNN', 'Consensus-GT', 'Consensus', 'Consensus-MVGNN']
CONS_MEMBERS = {'Consensus-GNN': ['gcn', 'gat', 'gin', 'sage'], 'Consensus-GT': ['graphormer', 'graphgps']}

UP = 'Up-sampled training set (actives duplicated to 1:1; val/test at natural ratio)'
D15 = '1:5 downsampled training set (all actives + 5 inactives per active)'
RUNS = [
    ('MMP HepG2 (AID 720637)', '5x repeated random 8:1:1', UP, 'Summary_1_models/upsampling_protocols/results_random_repeat', 'new'),
    ('MMP HepG2 (AID 720637)', '10-fold cross-validation', UP, 'Summary_1_models/upsampling_protocols/results_loo', 'new'),
    ('MMP HepG2 (AID 720637)', '5x scaffold 8:1:1', UP, 'Summary_1_models/upsampling_protocols/results_scaffold', 'new'),
    ('Caspase-3/7 CHO-K1 (AID 1347037)', '5x repeated random 8:1:1', UP, 'summary_1347037/upsampling_protocols/results_random_repeat', 'new'),
    ('Caspase-3/7 CHO-K1 (AID 1347037)', '10-fold cross-validation', UP, 'summary_1347037/upsampling/results_loo', 'chok1_old_loo'),
    ('Caspase-3/7 CHO-K1 (AID 1347037)', '5x scaffold 8:1:1', UP, 'summary_1347037/upsampling_protocols/results_scaffold', 'new'),
    ('Caspase-3/7 HepG2 (AID 1347034 only)', '5x repeated random 8:1:1', UP, 'summary_2_1437034/upsampling_1347034_random_repeat/results_random_repeat', 'new'),
    ('Caspase-3/7 HepG2 (AID 1347034 only)', '10-fold cross-validation', UP, 'summary_2_1437034/upsampling_1347034_protocols/results_loo', 'new'),
    ('Caspase-3/7 HepG2 (AID 1347034 only)', '5x scaffold 8:1:1', UP, 'summary_2_1437034/upsampling_1347034_protocols/results_scaffold', 'new'),
    ('Caspase-3/7 CHO-K1 (AID 1347037)', '5x scaffold 8:1:1', D15, 'summary_1347037/downsampling_1to5_scaffold/results_scaffold', 'new'),
]
GTC_CHOK1_UPLOO = 'graph_transformer_corrected/results/chok1/up_loo'


def summ_rows(path, name=None):
    """{(cpt): {metric: (mean, std)}} for one model from a *_summary.csv."""
    d = pd.read_csv(path)
    if name is not None:
        d = d[d['Model'] == name]
    out = {}
    for cpt in (0.5, 0.8):
        m = d[(d['CPT'] == cpt) & (d['Stat'] == 'mean')].iloc[0]
        s = d[(d['CPT'] == cpt) & (d['Stat'] == 'std')].iloc[0]
        out[cpt] = {k: (float(m[k]), float(s[k])) for k in METRICS}
    return out


def evaluate(y, p, cpt):
    pred = (p >= cpt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0
    spec = tn / (tn + fp) if tn + fp else 0
    return {'Sensitivity': sens * 100, 'Specificity': spec * 100, 'CCR': (sens + spec) / 2 * 100,
            'ROC-AUC': roc_auc_score(y, p), 'PR-AUC': average_precision_score(y, p),
            'MCC': matthews_corrcoef(y, pred), 'Bal-Acc': balanced_accuracy_score(y, pred) * 100,
            'F1': f1_score(y, pred, zero_division=0)}


def consensus_from_probs(res_dir, data_dir, unit, n, members):
    out = {}
    for cpt in (0.5, 0.8):
        rows = []
        for i in range(n):
            y = np.load(os.path.join(data_dir, f'{unit}_{i}', 'y_test.npy'))
            p = np.mean([np.load(os.path.join(res_dir, 'probs', f'{m}_{unit}{i}_test_probs.npy'))
                         for m in members], axis=0)
            rows.append(evaluate(y, p, cpt))
        df = pd.DataFrame(rows)
        out[cpt] = {k: (df[k].mean(), df[k].std()) for k in METRICS}
    return out


def collect(res_dir, kind):
    """Returns ordered list of (label, stats-or-note) for one run."""
    rows, d = [], os.path.join(HERE, res_dir)
    if kind == 'chok1_old_loo':
        g = os.path.join(HERE, GTC_CHOK1_UPLOO)
        for k, name in MODELS:
            if k in ('graphormer', 'graphgps'):
                rows.append((name, summ_rows(os.path.join(g, f'{name}_summary.csv'))))
            else:
                rows.append((name, summ_rows(os.path.join(d, f'{k}_loo_summary.csv'))))
        for c in CONS:
            rows.append((c, summ_rows(os.path.join(g, 'consensus_summary.csv'), c)))
        return rows
    for k, name in MODELS:
        p = os.path.join(d, f'{k}_summary.csv')
        rows.append((name, summ_rows(p) if os.path.exists(p) else None))
    cp = os.path.join(d, 'consensus_summary.csv')
    if os.path.exists(cp):
        rows += [(c, summ_rows(cp, c)) for c in CONS]
    else:  # consensus job not finished: compute the groups that do not need missing models
        data_dir = d.replace('results_', 'data_')
        unit, n = ('fold', 10) if res_dir.endswith('results_loo') else ('repeat', 5)
        for c in CONS:
            mem = CONS_MEMBERS.get(c)
            ok = mem and all(os.path.exists(os.path.join(d, 'probs', f'{m}_{unit}{i}_test_probs.npy'))
                             for m in mem for i in range(n))
            rows.append((c, consensus_from_probs(d, data_dir, unit, n, mem) if ok else None))
    return rows


def fmt(v, metric):
    m, s = v
    return f'{m:.1f} ± {s:.1f}' if metric in PCT else f'{m:.3f} ± {s:.3f}'


def table(rows, cpt):
    stats = [(lab, st[cpt] if st else None) for lab, st in rows]
    best = {k: max(round(st[k][0], 4) for _, st in stats if st) for k in METRICS}
    head = ['Model'] + METRICS
    body = []
    for lab, st in stats:
        if st is None:
            body.append([lab] + ['pending'] * len(METRICS))
            continue
        cells = [lab]
        for k in METRICS:
            c = fmt(st[k], k)
            cells.append(f'*{c}*' if round(st[k][0], 4) == best[k] else f' {c} ')
        body.append(cells)
    w = [max(len(r[i]) for r in [head] + body) for i in range(len(head))]
    line = lambda r: '  '.join(c.ljust(w[i]) if i == 0 else c.center(w[i]) for i, c in enumerate(r))
    sep = '  '.join('-' * x for x in w)
    out = [line(head), sep]
    for i, r in enumerate(body):
        if i == len(MODELS):
            out.append(sep)
        out.append(line(r))
    return '\n'.join(out)


def main():
    L = ['UP-SAMPLING AND 1:5 DOWNSAMPLING RESULTS -- SUMMARY ASSAYS',
         '=' * 60, '',
         'Values are test-set mean ± std across split units (5 repeats or 10 folds).',
         'In each column the best mean is wrapped in asterisks, e.g. *0.812 ± 0.040*;',
         'ties are all marked. All metrics are higher-is-better.',
         'Sensitivity, Specificity, CCR and Bal-Acc are percentages (CCR = Bal-Acc).',
         'ROC-AUC and PR-AUC do not depend on the cutoff; the threshold metrics are',
         'shown at a 0.5 and a 0.8 probability cutoff.',
         'Graphormer and GraphGPS are the corrected architectures in every table.',
         'Consensus = mean of the 9 models other than MV-GNN; Consensus-MVGNN adds MV-GNN.',
         '"pending" = that model had not finished when the results were copied.', '']
    for i, (assay, proto, bal, res, kind) in enumerate(RUNS, 1):
        rows = collect(res, kind)
        L += ['', '#' * 100, f'{i}. {assay}', f'   Split:    {proto}', f'   Training: {bal}',
              f'   Source:   {res}' + (f'  (+ corrected GT rows from {GTC_CHOK1_UPLOO})' if kind == 'chok1_old_loo' else ''),
              '#' * 100]
        if any(st is None for _, st in rows):
            L.append('   NOTE: SVM job unfinished (8 of 10 folds copied), so SVM, Consensus-Classical,\n'
                     '   Consensus and Consensus-MVGNN are pending. Consensus-GNN and Consensus-GT\n'
                     '   were computed here from the saved per-fold predictions.')
        for cpt in (0.5, 0.8):
            L += ['', f'Cutoff {cpt}', table(rows, cpt)]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
