"""LaTeX tables (same format as acs-si-2.tex) for the up-sampled runs on the
three summary assays and the CHO-K1 1:5 scaffold run.

Format: the SI table layout (multirow model names, CPT 0.5/0.8 rows,
ROC-AUC/PR-AUC shown once per model, same model
groups plus one MV-GNN-cross row between the GNNs and the graph
transformers; MV-GNN-cross is not part of any consensus), with percentages to 2 decimals and all other values to 3, and in each column, the best value
in bold and the second best underlined (build_table below).

95% CI on ROC-AUC and PR-AUC: the Nadeau-Bengio corrected resampled
t-interval used throughout the SI (compute_ci_significance.corrected_ci):
    mean +/- t(0.975, n-1) * sqrt((1/n + n_test/n_train) * var)
over the per-repeat / per-fold test values. Unlike the SI script, which had
to approximate n_train/n_test because the split arrays were not available
locally, the real sizes are read from the split files here:
    n_test  = mean test-set size across units
    n_train = mean number of distinct training compounds across units: for
              up-sampled runs N - n_val - n_test (the duplicated actives add
              no new compounds); for the 1:5 run the downsampled training set.
Every per-unit metric is recomputed from the saved test-set probabilities and
true labels (the per-unit CSVs round percentages to 0.1, too coarse for two
decimals); threshold metrics are then the mean across units, as in the SI.


Usage: python build_upsampling_1to5_tex.py
Output: tables_upsampling_1to5.tex
"""
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score, f1_score, confusion_matrix)

import table_helpers as th

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'tables_upsampling_1to5.tex')
METRICS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
FILE_KEYS = [('rf', 'Random Forest'), ('svm', 'SVM'), ('xgb', 'XGBoost'), ('gcn', 'GCN'),
             ('gat', 'GAT'), ('gin', 'GIN'), ('sage', 'SAGE'), ('mvgnn', 'MV-GNN-cross'),
             ('graphormer', 'Graphormer'), ('graphgps', 'GraphGPS')]
# Row groups: the SI groups with MV-GNN-cross as its own row between the GNNs
# and the graph transformers. It is NOT a member of any consensus (CONS_MEMBERS).
GROUPS = [th.GROUPS[0], th.GROUPS[1], ('Multi-View Graph Neural Network', ['MV-GNN-cross'])] + list(th.GROUPS[2:])
CONS = ['Consensus-Classical', 'Consensus-GNN', 'Consensus-GT', 'Consensus']
GTC_UPLOO = os.path.join(HERE, 'graph_transformer_corrected', 'results', 'chok1', 'up_loo')
UP = 'random up-sampling (1:1) of train'
D15 = '1:5 downsampled train (all actives, 5 inactives per active)'

# (assay caption, label, split text, unit text, balancing, results dir, data dir, kind)
TABLES = [
    ('the MMP summary assay (AID 720637)', 'tab:mmp_up_random', 'repeated 80/10/10 split', 'mean across 5 repeats', UP,
     'Summary_1_models/upsampling_protocols/results_random_repeat', 'Summary_1_models/upsampling_protocols/data_random_repeat', 'new'),
    ('the MMP summary assay (AID 720637)', 'tab:mmp_up_loo', '10-fold cross-validation', 'mean across 10 folds', UP,
     'Summary_1_models/upsampling_protocols/results_loo', 'Summary_1_models/upsampling_protocols/data_loo', 'new'),
    ('the MMP summary assay (AID 720637)', 'tab:mmp_up_scaffold', 'scaffold split, 5 repeats', 'mean across 5 repeats', UP,
     'Summary_1_models/upsampling_protocols/results_scaffold', 'Summary_1_models/upsampling_protocols/data_scaffold', 'new'),
    ('the Caspase-3/7 CHO-K1 assay (AID 1347037)', 'tab:chok1_up_random', 'repeated 80/10/10 split', 'mean across 5 repeats', UP,
     'summary_1347037/upsampling_protocols/results_random_repeat', 'summary_1347037/upsampling_protocols/data_random_repeat', 'new'),
    ('the Caspase-3/7 CHO-K1 assay (AID 1347037)', 'tab:chok1_up_loo', '10-fold cross-validation', 'mean across 10 folds', UP,
     'summary_1347037/upsampling/results_loo', os.environ.get('CHOK1_UPLOO_DATA', ''), 'chok1_old_loo'),
    ('the Caspase-3/7 CHO-K1 assay (AID 1347037)', 'tab:chok1_up_scaffold', 'scaffold split, 5 repeats', 'mean across 5 repeats', UP,
     'summary_1347037/upsampling_protocols/results_scaffold', 'summary_1347037/upsampling_protocols/data_scaffold', 'new'),
    ('the Caspase-3/7 CHO-K1 assay (AID 1347037)', 'tab:chok1_1to5_scaffold', 'scaffold split, 5 repeats', 'mean across 5 repeats', D15,
     'summary_1347037/downsampling_1to5_scaffold/results_scaffold',
     'summary_1347037/downsampling_1to5_scaffold/data_scaffold', 'new'),
    ('the Caspase-3/7 HepG2 assay (AID 1347034 only)', 'tab:hepg2only_up_random', 'repeated 80/10/10 split', 'mean across 5 repeats', UP,
     'summary_2_1437034/upsampling_1347034_random_repeat/results_random_repeat',
     'summary_2_1437034/upsampling_1347034_random_repeat/data_random_repeat', 'new'),
    ('the Caspase-3/7 HepG2 assay (AID 1347034 only)', 'tab:hepg2only_up_loo', '10-fold cross-validation', 'mean across 10 folds', UP,
     'summary_2_1437034/upsampling_1347034_protocols/results_loo',
     'summary_2_1437034/upsampling_1347034_protocols/data_loo', 'new'),
    ('the Caspase-3/7 HepG2 assay (AID 1347034 only)', 'tab:hepg2only_up_scaffold', 'scaffold split, 5 repeats', 'mean across 5 repeats', UP,
     'summary_2_1437034/upsampling_1347034_protocols/results_scaffold',
     'summary_2_1437034/upsampling_1347034_protocols/data_scaffold', 'new'),
]


CONS_MEMBERS = {
    'Consensus-Classical': ['rf', 'svm', 'xgb'],
    'Consensus-GNN': ['gcn', 'gat', 'gin', 'sage'],
    'Consensus-GT': ['graphormer', 'graphgps'],
    'Consensus': ['rf', 'svm', 'xgb', 'gcn', 'gat', 'gin', 'sage', 'graphormer', 'graphgps'],
}


def unit_metrics(y, p, cpt):
    """Exact (unrounded) metrics for one split unit -- the per-unit CSVs store
    percentages rounded to 0.1, too coarse for a two-decimal mean."""
    pred = (p >= cpt).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {'Sensitivity': sens * 100, 'Specificity': spec * 100, 'CCR': (sens + spec) / 2 * 100,
            'ROC-AUC': roc_auc_score(y, p), 'PR-AUC': average_precision_score(y, p),
            'MCC': matthews_corrcoef(y, pred), 'Bal-Acc': balanced_accuracy_score(y, pred) * 100,
            'F1': f1_score(y, pred, zero_division=0)}


def load_run(res, data, kind):
    """{display name: {cpt: DataFrame (units x METRICS)}} recomputed from each
    unit's saved test-set probabilities and true labels. Consensus groups are
    the unweighted mean of their members' probabilities (as compute_consensus.py).
    CHO-K1 CV (kind 'chok1_old_loo'): RF/SVM/XGB/GNNs from the original run,
    Graphormer/GraphGPS from the corrected retrain."""
    res = os.path.join(HERE, res)
    data = data if os.path.isabs(data) else os.path.join(HERE, data)
    unit = 'fold' if res.rstrip('/').endswith('results_loo') else 'repeat'
    n = len([d for d in os.listdir(data) if d.startswith(unit + '_')])
    gt_name = {'graphormer': 'Graphormer', 'graphgps': 'GraphGPS'}

    def probs(key, i):
        if kind == 'chok1_old_loo' and key in gt_name:
            return np.load(os.path.join(GTC_UPLOO, 'probs', f'{gt_name[key]}_{i}_test_probs.npy'))
        return np.load(os.path.join(res, 'probs', f'{key}_{unit}{i}_test_probs.npy'))

    out = {}
    labels = [np.load(os.path.join(data, f'{unit}_{i}', 'y_test.npy')) for i in range(n)]
    entries = [(name, [k]) for k, name in FILE_KEYS] + [(c, CONS_MEMBERS[c]) for c in CONS]
    for name, members in entries:
        rows = {0.5: [], 0.8: []}
        for i in range(n):
            p = np.mean([probs(m, i) for m in members], axis=0)
            assert len(p) == len(labels[i]), (res, name, i)
            for cpt in (0.5, 0.8):
                rows[cpt].append(unit_metrics(labels[i], p, cpt))
        out[name] = {cpt: pd.DataFrame(rows[cpt])[METRICS] for cpt in rows}
    return out


N_COMPOUNDS = {  # Active + Inactive compounds after cleaning (same filter as the split scripts)
    'the MMP summary assay (AID 720637)': 5937,
    'the Caspase-3/7 CHO-K1 assay (AID 1347037)': 7496,
    'the Caspase-3/7 HepG2 assay (AID 1347034 only)': 7271,
}


def split_sizes(data_dir, assay, bal):
    """(n_train, n_test, n_units) averaged over the units on disk.
    Up-sampled runs: every compound outside val/test is in the training pool
    exactly once before duplication, so n_train = N - n_val - n_test.
    1:5 run: the downsampled training set has no duplicates, so its size."""
    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(f'split data dir not found: {data_dir!r} (set CHOK1_UPLOO_DATA for the CHO-K1 CV run)')
    units = sorted(d for d in os.listdir(data_dir) if d.startswith(('repeat_', 'fold_')))
    if not units:
        raise FileNotFoundError(f'no repeat_/fold_ dirs in {data_dir}')
    tr, te = [], []
    for u in units:
        n_val = len(np.load(os.path.join(data_dir, u, 'y_val.npy')))
        n_te = len(np.load(os.path.join(data_dir, u, 'y_test.npy')))
        if bal == UP:
            tr.append(N_COMPOUNDS[assay] - n_val - n_te)
        else:
            tr.append(len(np.load(os.path.join(data_dir, u, 'y_train.npy'))))
        te.append(n_te)
    return float(np.mean(tr)), float(np.mean(te)), len(units)


def corrected_ci(vals, n_train, n_test):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    mean = vals.mean()
    var = vals.var(ddof=1)
    se = np.sqrt((1.0 / n + n_test / n_train) * var)
    moe = stats.t.ppf(0.975, df=n - 1) * se
    return mean - moe, mean + moe





DEC = 3  # ROC-AUC, PR-AUC, MCC, F1
PCT_DEC = 2  # Sensitivity, Specificity, CCR, Bal-Acc (percentages)
PCT_COLS = ('Sensitivity', 'Specificity', 'CCR', 'Bal-Acc')


def ndec(c):
    return PCT_DEC if c in PCT_COLS else DEC


def build_table(caption, label, rows_by_model, roc_ci_map, pr_ci_map):
    """Same layout as table_helpers.build_table (SI format), except that
    percentages are shown to 2 decimals and all other values to 3, and each column marks the best value in bold
    and the second-best value underlined. Ranking uses the DISPLAYED (rounded)
    values, so cells that print identically are always marked identically."""
    present = [(g, [m for m in ms if m in rows_by_model]) for g, ms in GROUPS]
    present = [(g, ms) for g, ms in present if ms]

    def disp_val(c, m, cpt):
        r = rows_by_model[m]
        return round(r[0.5][c] if c in ('ROC-AUC', 'PR-AUC') else r[cpt][c], ndec(c))

    rank = {}
    for c in METRICS:
        vals = set()
        for _, ms in present:
            for m in ms:
                for cpt in ((0.5,) if c in ('ROC-AUC', 'PR-AUC') else (0.5, 0.8)):
                    vals.add(disp_val(c, m, cpt))
        top = sorted(vals, reverse=True)
        rank[c] = (top[0], top[1] if len(top) > 1 else None)

    def mark(c, v, text, math=False):
        best, second = rank[c]
        if v == best:
            return f'{{\\boldmath${text}$}}' if math else f'\\textbf{{{text}}}'
        if second is not None and v == second:
            return f'\\underline{{${text}$}}' if math else f'\\underline{{{text}}}'
        return f'${text}$' if math else text

    def auc(c, m, ci):
        v = disp_val(c, m, 0.5)
        lo, hi = ci
        return mark(c, v, f'{v:.{DEC}f} \\pm {(hi - lo) / 2:.{DEC}f}', math=True)

    L = ['\\begin{table*}[t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
         '\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{llcccccccc}', '\\toprule',
         '\\textbf{Model} & \\textbf{CPT} & \\textbf{Sensitivity} & \\textbf{Specificity} & '
         '\\textbf{CCR} & \\textbf{ROC-AUC (95\\% CI)} & \\textbf{PR-AUC (95\\% CI)} & \\textbf{MCC} & '
         '\\textbf{Bal-Acc} & \\textbf{F1} \\\\',
         ' & & (\\%) & (\\%) & (\\%) & & & & (\\%) & \\\\', '\\midrule']
    for gi, (g, ms) in enumerate(present):
        L += [f'\\multicolumn{{10}}{{l}}{{\\textit{{{g}}}}} \\\\', '\\midrule']
        for m in ms:
            roc_s, pr_s = auc('ROC-AUC', m, roc_ci_map[m]), auc('PR-AUC', m, pr_ci_map[m])
            for i, cpt in enumerate((0.5, 0.8)):
                cells = [mark(c, disp_val(c, m, cpt), f'{disp_val(c, m, cpt):.{ndec(c)}f}')
                         for c in ('Sensitivity', 'Specificity', 'CCR')]
                tail = [mark(c, disp_val(c, m, cpt), f'{disp_val(c, m, cpt):.{ndec(c)}f}')
                        for c in ('MCC', 'Bal-Acc', 'F1')]
                name = f'\\multirow{{2}}{{*}}{{{th.MODEL_DISPLAY.get(m, m)}}}' if i == 0 else ''
                rc = f'\\multirow{{2}}{{*}}{{{roc_s}}}' if i == 0 else ''
                pc = f'\\multirow{{2}}{{*}}{{{pr_s}}}' if i == 0 else ''
                L.append(f'{name} & {cpt} & ' + ' & '.join(cells) + f' & {rc} & {pc} & ' + ' & '.join(tail) + ' \\\\')
        if gi < len(present) - 1:
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}%', '}', '\\end{table*}', '']
    return '\n'.join(L)


def main():
    parts = {}
    for t in TABLES:
        parts[t[0]] = parts.get(t[0], 0) + 1
    seen, blocks, log = {}, [], []
    for assay, label, split, unit_txt, bal, res, data, kind in TABLES:
        runs = load_run(res, data, kind)
        n_train, n_test, n_units = split_sizes(os.path.join(HERE, data) if not os.path.isabs(data) else data, assay, bal)
        rows, roc_ci, pr_ci = {}, {}, {}
        for name, pu in runs.items():
            assert len(pu[0.5]) == n_units, (label, name, len(pu[0.5]), n_units)
            rows[name] = {cpt: {m: float(pu[cpt][m].mean()) for m in METRICS} for cpt in (0.5, 0.8)}
            roc_ci[name] = corrected_ci(pu[0.5]['ROC-AUC'], n_train, n_test)
            pr_ci[name] = corrected_ci(pu[0.5]['PR-AUC'], n_train, n_test)
        key = assay
        seen[key] = seen.get(key, 0) + 1
        part = f'\\textit{{(Part {seen[key]} of {parts[key]})}} ' if parts[key] > 1 else ''
        cap = (f'{part}Performance on {assay} -- {split}, {bal} ({unit_txt}). '
               f'ROC-AUC and PR-AUC are mean $\\pm$ 95\\% CI half-width (Nadeau--Bengio corrected '
               f'resampled $t$-interval, $n={n_units}$); other columns are means. MV-GNN-cross is shown for reference and is not part of any consensus. Percentages to two decimals, all other values to three; in each column the best value is in bold and the second best is underlined.')
        blocks.append(build_table(cap, label, rows, roc_ci, pr_ci))
        log.append(f'{label:28s} units={n_units:2d} n_train={n_train:7.1f} n_test={n_test:6.1f}')
    header = ('% Generated by build_upsampling_1to5_tex.py -- up-sampled runs on the three summary\n'
              '% assays + CHO-K1 1:5 scaffold run. Same format as acs-si-2.tex.\n'
              '\n')
    body = '\n'.join(blocks)
    open(OUT, 'w').write(header + body)
    print('\n'.join(log))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
