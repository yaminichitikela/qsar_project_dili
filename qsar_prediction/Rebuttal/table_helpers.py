"""
Builds a single .tex file with all requested results tables, in the format
given by the user's pasted example (multirow model name, CPT 0.5/0.8 rows),
with two corrections made after review:
  1. No row highlighting. The Consensus (Full) row is no longer unconditionally
     shaded -- that implied it was always the top performer even in tables
     where a different model/group actually scored higher on a given metric.
     The best value in every column is marked by bolding only, computed
     per-column across the whole table, so whichever row is genuinely best
     is what stands out.
  2. Significant-digit reporting (Editor comment E4) is done via the Wheeler
     (2020) "Level of Precision" rule -- the same rule, and the same
     wheeler_precision()/round_to_sig_figs() functions, already used in
     compute_ci_significance.py elsewhere in this project: round the margin
     of error to 1-2 significant figures, then round the mean to that same
     decimal place, so e.g. 0.6356 +/- 0.0484 is reported as 0.64 +/- 0.05,
     not with digits the data doesn't support. Applied to ROC-AUC and PR-AUC
     (the two metrics this project already computes real uncertainty for).
     MCC and F1 are reported to a fixed 3 decimals (no CI chain exists for
     them); Sensitivity/Specificity/CCR/Bal-Acc stay at 1 decimal (percentages
     are already at reasonable precision).

Per Reviewer 2 comment 5 ("Provide only single ROC-AUC and PR-AUC value for
different CPT thresholds"), ROC-AUC and PR-AUC are shown ONCE per model
(multirow spanning both CPT rows), not repeated/duplicated with different
threshold-irrelevant values -- unlike the user's pasted example, which still
repeated them. Threshold-dependent metrics (Sensitivity, Specificity, CCR,
MCC, Bal-Acc, F1) remain per-CPT-row since they genuinely differ by threshold.

Per Editor comment E2, every table's ROC-AUC/PR-AUC columns carry a 95% CI:
  - For LOO / Scaffold / Up-sampling-LOO tables: the already-verified
    Nadeau-Bengio corrected t-interval, pulled from the CI_Summary sheet of
    QSAR_Rebuttal_All_Results.xlsx (both metrics).
  - For single 80/10/10 "Random" split tables (and the two DILI tables,
    which are also single fixed evaluations): a bootstrap 95% CI (2000
    resamples, percentile method) computed directly here from each model's
    saved test-set probabilities + true labels, wherever those files are
    available locally. Where the raw probability/label files are not
    present on this machine (flagged explicitly, not guessed), that cell
    falls back to a plain 3-decimal value with no implied uncertainty rather
    than a fabricated interval.

MV-GNN-cross is excluded from every per-dataset table (not "our" architecture,
a baseline comparison model) except one dedicated comparison table, matching
the already-established G6 Table 3 structure (Consensus vs MV-GNN-cross vs
Consensus-MVGNN, across all 4 datasets). A second dedicated table reproduces
the Xu et al. replication comparison (G6 Table 4), DILI only.

Usage: python build_rebuttal_latex_tables.py
Output: rebuttal_results_tables.tex
"""
import os
import numpy as np
import openpyxl

BASE = os.path.dirname(__file__)
WB_PATH = os.path.join(BASE, 'QSAR_Rebuttal_All_Results.xlsx')
OUT_PATH = os.path.join(BASE, 'rebuttal_results_tables.tex')

N_BOOT = 2000
CI_DECIMALS = 3
BOOT_SEED = 42

MODEL_DISPLAY = {
    'Random Forest': 'Random Forest', 'RF': 'Random Forest', 'RF-CV': 'Random Forest',
    'SVM': 'SVM', 'SVM-CV': 'SVM',
    'XGBoost': 'XGBoost', 'XGB': 'XGBoost', 'XGB-CV': 'XGBoost',
    'GCN': 'GCN', 'GAT': 'GAT', 'GIN': 'GIN', 'SAGE': 'SAGE',
    'Graphormer': 'Graphormer', 'GraphGPS': 'GraphGPS',
    'Consensus-Classical': 'Consensus-Classical',
    'Consensus-GNN': 'Consensus-GNN',
    'Consensus-GT': 'Consensus-GT',
    'Consensus': 'Consensus (Full)',
}
GROUPS = [
    ('Classical Machine Learning', ['Random Forest', 'SVM', 'XGBoost']),
    ('Graph Neural Networks', ['GCN', 'GAT', 'GIN', 'SAGE']),
    ('Graph Transformers', ['Graphormer', 'GraphGPS']),
    ('Consensus Ensembles', ['Consensus-Classical', 'Consensus-GNN', 'Consensus-GT', 'Consensus']),
]
NUMERIC_COLS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']


def load_sheet(sheet_name):
    wb = openpyxl.load_workbook(WB_PATH, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    header, data = rows[0], rows[1:]
    return [dict(zip(header, r)) for r in data]


import json
MISSING_CI_PATH = os.path.join(BASE, 'missing_consensus_ci.json')
MISSING_CI = json.load(open(MISSING_CI_PATH)) if os.path.exists(MISSING_CI_PATH) else {}


def ci_from_summary(dataset_label, method_label, model_key, metric='ROC-AUC', fallback_key=None, cache={}):
    key = 'ci_summary'
    if key not in cache:
        cache[key] = load_sheet('CI_Summary')
    rows = cache[key]
    for r in rows:
        if (r['Dataset'] == dataset_label and r['Method'] == method_label
                and r['Model'] == model_key and r['Metric'] == metric):
            lo, hi = r['CI95_Low'], r['CI95_High']
            return max(lo, 0.0), min(hi, 1.0)
    if fallback_key and fallback_key in MISSING_CI and model_key in MISSING_CI[fallback_key]:
        entry = MISSING_CI[fallback_key][model_key]
        # entry is either [lo, hi] (legacy, ROC-AUC only) or {"ROC-AUC": [lo,hi], "PR-AUC": [lo,hi]}
        if isinstance(entry, dict):
            if metric in entry:
                lo, hi = entry[metric]
                return max(lo, 0.0), min(hi, 1.0)
            return None
        if metric == 'ROC-AUC':
            lo, hi = entry
            return max(lo, 0.0), min(hi, 1.0)
    return None


def bootstrap_ci_metric(y_true, y_prob, metric='ROC-AUC', n_boot=N_BOOT, seed=BOOT_SEED):
    from sklearn.metrics import roc_auc_score, average_precision_score
    metric_fn = roc_auc_score if metric == 'ROC-AUC' else average_precision_score
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(metric_fn(yt, yp))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return max(lo, 0.0), min(hi, 1.0)


# Backward-compatible alias (ROC-AUC only), used by earlier scripts.
def bootstrap_ci_auc(y_true, y_prob, n_boot=N_BOOT, seed=BOOT_SEED):
    return bootstrap_ci_metric(y_true, y_prob, 'ROC-AUC', n_boot, seed)


def round_to_sig_figs(x, sig_figs):
    if x == 0:
        return 0.0, 0
    decimal_place = sig_figs - int(np.floor(np.log10(abs(x)))) - 1
    return round(x, decimal_place), decimal_place


def wheeler_precision(mean, moe):
    """Wheeler (2020) 'Level of Precision' rule: round the margin of error to
    1 sig fig (or 2, if its leading digit is 1 or 2), then round the mean to
    that same decimal place -- same function already used project-wide in
    compute_ci_significance.py."""
    if moe == 0:
        return mean, moe
    leading_digit = int(f'{moe:.1e}'[0])
    sig_figs = 2 if leading_digit in (1, 2) else 1
    rounded_moe, decimal_place = round_to_sig_figs(moe, sig_figs)
    rounded_mean = round(mean, decimal_place)
    return rounded_mean, rounded_moe


def fmt_wheeler(point_est, ci, decimals_fallback=3):
    """Formats a value with Wheeler precision-matched mean +/- moe when a CI
    is available; falls back to a plain fixed-decimal value (no implied
    uncertainty) when no CI could be computed for this cell. Returns
    (display_string, rounded_mean) -- the rounded mean is what "best value in
    this column" comparisons should use, not the raw point estimate, so two
    cells that display identically (e.g. both "0.186" after rounding) are
    never bolded differently based on a sub-precision difference the reader
    can't actually see."""
    if ci is None:
        if point_est is None:
            return '--', None
        rounded = round(point_est, decimals_fallback)
        return f'{rounded:.{decimals_fallback}f}', rounded
    lo, hi = ci
    moe = (hi - lo) / 2.0
    # Fixed 3 decimals for both mean and margin (E4 response: "ROC-AUC and
    # PR-AUC are given to three decimal places"), so models whose CIs overlap
    # can still be told apart in the table.
    mean_r, moe_r = round(point_est, CI_DECIMALS), round(moe, CI_DECIMALS)
    return f'{mean_r:.{CI_DECIMALS}f} \\pm {moe_r:.{CI_DECIMALS}f}', mean_r


def best_mask(values):
    """values: list of floats (None allowed); returns bool list for argmax."""
    valid = [v for v in values if v is not None]
    if not valid:
        return [False] * len(values)
    m = max(valid)
    return [v is not None and abs(v - m) < 1e-9 for v in values]


def build_table(caption, label, dataset_label_ci, method_label_ci, rows_by_model,
                 roc_ci_map, pr_ci_map=None):
    """rows_by_model: {model_key: {0.5: row_dict_test, 0.8: row_dict_test}}
    roc_ci_map / pr_ci_map: {model_key: (lo, hi) or None}. No row highlighting
    -- the best value in each column is marked by bolding only (dropped the
    unconditional Consensus-row highlight, since it could mark a row as if it
    were the best performer even when a different model/group actually scored
    higher on a given metric in that table)."""
    pr_ci_map = pr_ci_map or {}

    # Bolding compares DISPLAYED (rounded) values, not raw ones -- otherwise
    # two cells that render identically (e.g. both "0.186" after 3-decimal
    # rounding) could be bolded differently based on a sub-precision gap the
    # reader can't see. So: round every cell first, then find the best among
    # those rounded values, then bold every cell matching that rounded best
    # (ties are bolded together, not arbitrarily broken).
    rounded_vals = {c: [] for c in NUMERIC_COLS}
    for _, members in GROUPS:
        for m in members:
            if m not in rows_by_model:
                continue
            for cpt in (0.5, 0.8):
                r = rows_by_model[m].get(cpt)
                if not r:
                    continue
                for c in ('Sensitivity', 'Specificity', 'CCR', 'Bal-Acc'):
                    if r[c] is not None:
                        rounded_vals[c].append(round(r[c], 1))
                for c, dec in (('MCC', 3), ('F1', 3)):
                    if r[c] is not None:
                        rounded_vals[c].append(round(r[c], dec))
            r05 = rows_by_model[m].get(0.5)
            r08 = rows_by_model[m].get(0.8)
            roc = r05['ROC-AUC'] if r05 else (r08['ROC-AUC'] if r08 else None)
            pr = r05['PR-AUC'] if r05 else (r08['PR-AUC'] if r08 else None)
            if roc is not None:
                _, roc_r = fmt_wheeler(roc, roc_ci_map.get(m))
                if roc_r is not None:
                    rounded_vals['ROC-AUC'].append(roc_r)
            if pr is not None:
                _, pr_r = fmt_wheeler(pr, pr_ci_map.get(m))
                if pr_r is not None:
                    rounded_vals['PR-AUC'].append(pr_r)
    best = {c: (max(rounded_vals[c]) if rounded_vals[c] else None) for c in NUMERIC_COLS}

    def isbest(c, v_rounded):
        return v_rounded is not None and best[c] is not None and abs(v_rounded - best[c]) < 1e-9

    def fmt_pct_or_mcc(c, v, decimals=1):
        if v is None:
            return '--'
        v_r = round(v, decimals)
        s = f'{v_r:.{decimals}f}'
        return f'\\textbf{{{s}}}' if isbest(c, v_r) else s

    def fmt_auc_col(c, v, ci):
        if v is None:
            return '--'
        s, v_r = fmt_wheeler(v, ci)
        s = f'${s}$' if ci is not None else s
        return f'\\textbf{{{s}}}' if isbest(c, v_r) else s

    lines = []
    lines.append('\\begin{table*}[t]')
    lines.append('\\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{label}}}')
    lines.append('\\resizebox{\\textwidth}{!}{%')
    lines.append('\\begin{tabular}{llcccccccc}')
    lines.append('\\toprule')
    lines.append('\\textbf{Model} & \\textbf{CPT} & \\textbf{Sensitivity} & \\textbf{Specificity} & '
                 '\\textbf{CCR} & \\textbf{ROC-AUC (95\\% CI)} & \\textbf{PR-AUC (95\\% CI)} & \\textbf{MCC} & '
                 '\\textbf{Bal-Acc} & \\textbf{F1} \\\\')
    lines.append(' & & (\\%) & (\\%) & (\\%) & & & & (\\%) & \\\\')
    lines.append('\\midrule')

    for group_name, members in GROUPS:
        present = [m for m in members if m in rows_by_model]
        if not present:
            continue
        lines.append(f'\\multicolumn{{10}}{{l}}{{\\textit{{{group_name}}}}} \\\\')
        lines.append('\\midrule')
        for m in present:
            disp = MODEL_DISPLAY.get(m, m)
            r05 = rows_by_model[m].get(0.5)
            r08 = rows_by_model[m].get(0.8)
            roc = r05['ROC-AUC'] if r05 else (r08['ROC-AUC'] if r08 else None)
            pr = r05['PR-AUC'] if r05 else (r08['PR-AUC'] if r08 else None)
            roc_s = fmt_auc_col('ROC-AUC', roc, roc_ci_map.get(m))
            pr_s = fmt_auc_col('PR-AUC', pr, pr_ci_map.get(m))

            for i, cpt in enumerate((0.5, 0.8)):
                r = rows_by_model[m].get(cpt)
                if r is None:
                    continue
                sens = fmt_pct_or_mcc('Sensitivity', r['Sensitivity'])
                spec = fmt_pct_or_mcc('Specificity', r['Specificity'])
                ccr = fmt_pct_or_mcc('CCR', r['CCR'])
                mcc = fmt_pct_or_mcc('MCC', r['MCC'], decimals=3)
                bacc = fmt_pct_or_mcc('Bal-Acc', r['Bal-Acc'])
                f1 = fmt_pct_or_mcc('F1', r['F1'], decimals=3)
                model_cell = f'\\multirow{{2}}{{*}}{{{disp}}}' if i == 0 else ''
                roc_cell = f'\\multirow{{2}}{{*}}{{{roc_s}}}' if i == 0 else ''
                pr_cell = f'\\multirow{{2}}{{*}}{{{pr_s}}}' if i == 0 else ''
                lines.append(
                    f'{model_cell} & {cpt} & {sens} & {spec} & {ccr} & '
                    f'{roc_cell} & {pr_cell} & {mcc} & {bacc} & {f1} \\\\'
                )
        lines.append('\\midrule' if group_name != GROUPS[-1][0] else '')
    if lines[-1] == '\\midrule' or lines[-1] == '':
        lines.pop()
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}%')
    lines.append('}')
    lines.append('\\end{table*}')
    lines.append('')
    return '\n'.join(lines)


def rows_from_sheet(sheet_rows, dataset_filter=None, method_filter=None, stage_filter=None,
                     row_type_filter='Test'):
    out = {}
    for r in sheet_rows:
        if dataset_filter and r.get('Dataset') != dataset_filter:
            continue
        if method_filter and r.get('Method') != method_filter:
            continue
        if stage_filter and r.get('Stage') != stage_filter:
            continue
        if 'RowType' in r and r['RowType'] != row_type_filter:
            continue
        model = r['Model']
        cpt = r['CPT']
        out.setdefault(model, {})[cpt] = {c: r[c] for c in NUMERIC_COLS}
    return out



def load_npy(path):
    return np.load(path) if os.path.exists(path) else None
