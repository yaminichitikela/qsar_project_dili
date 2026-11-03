"""Computes 95% confidence intervals and paired significance tests for every
model, using ONLY the repeated-evaluation results that already exist under
each assay's results_loo/ (10-fold) and results_scaffold/ (5-repeat)
directories. No retraining -- this is a pure post-hoc analysis pass, run to
satisfy the Editor's two explicit conditions (confidence intervals; paired
significance tests) on the existing rebuttal results.

Statistical method (and why it isn't a naive paired t-test, and why it isn't
Wheeler (2020)'s independent two-sample t-test either)
-------------------------------------------------------------
Folds of a k-fold CV (or repeats of a resplit) share overlapping training
data, so the per-fold/per-repeat values are NOT independent. A plain paired
t-test on them understates the true variance and inflates the false-positive
rate -- shown empirically by:
    Dietterich, T. G. (1998). "Approximate Statistical Tests for Comparing
    Supervised Classification Learning Algorithms." Neural Computation,
    10(7), 1895-1923.

The Editor's cited reference (Wheeler, 2020, "How to Calculate a Confidence
Interval") gives the correct formula for two INDEPENDENT samples (its Eq.
2/3, Welch's t-test) -- but our two "samples" per comparison are each
model's scores on the SAME folds (same held-out compounds every time), so
they are paired/correlated, not independent. Wheeler's document is a
general engineering primer and does not cover the repeated-CV case. Using
its independent-samples formula here would itself be a methodological
mismatch, just a different one than the naive paired t-test above.

We instead use the corrected resampled t-test of:
    Nadeau, C. & Bengio, Y. (2003). "Inference for the Generalization
    Error." Machine Learning, 52, 239-281.
which inflates the variance of the mean (of a single series, or of a paired
difference) by (1/n + n_test/n_train) instead of the naive 1/n, to account
for the correlation induced by shared training data across folds/repeats.
This corrected variance is used both for:
  (a) the 95% CI of a single model's repeated-evaluation metric --
      mean +/- t(0.975, df=n-1) * sqrt((1/n + n_test/n_train) * sample_var)
  (b) the corrected paired t-test between two models evaluated on the same
      folds/repeats (same formula, applied to the per-fold differences).

A Wilcoxon signed-rank test is reported alongside as a non-parametric
cross-check (recommended for two-classifier comparisons by:
    Demsar, J. (2006). "Statistical Comparisons of Classifiers over
    Multiple Data Sets." Journal of Machine Learning Research, 7, 1-30.
-- note Demsar's own setting is comparing two classifiers ACROSS several
datasets, one paired value per dataset; here the paired values are instead
per fold/repeat WITHIN one dataset, which is the Dietterich/Nadeau-Bengio
setting above -- Wilcoxon is still a valid, assumption-light test for a set
of paired differences either way, so it is kept as a cross-check). Its
two-sided p-value floor is 1/2^n (0.0625 at n=5, ~0.001 at n=10) -- at n=5
(scaffold) it structurally cannot report p<0.05 regardless of effect size,
so the corrected t-test is the primary test and Wilcoxon is supporting
evidence, not the deciding one.

p-values within each (assay, split-scheme) family of 3 comparisons are
Holm-Bonferroni corrected:
    Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test
    Procedure." Scandinavian Journal of Statistics, 6(2), 65-70.

Significant-digit rounding follows Wheeler (2020)'s "Level of Precision"
rule even though its significance-test formula isn't used here: round the
margin of error to 1 significant digit, or 2 if its leading digit is 1 or
2, then round the mean/difference to the same decimal place (matching
Wheeler's own example: 131.773 +/- 2.4329 -> 131.8 +/- 2.4). That rounding
convention is independent of which variance estimator produced the margin
of error, so it's applied here too.

n_test / n_train approximation
--------------------------------
The data_loo/ and data_scaffold/ directories that held the actual per-fold
split arrays were only ever materialized on the training cluster and are
not present in this local copy of the repo, so exact per-fold train/test
counts cannot be read back. They are instead approximated from the
documented split protocol (prepare_loo_folds.py: 10 folds, 8 for training;
prepare_scaffold_splits.py: 8:1:1 target ratio, both cases downsampled to
1:1 on the training portion only) and the assay's actual active/inactive
counts, read fresh from the cleaned CSV (same source and filter
check_pos_neg_ratio.py uses -- Active/Inactive rows only, Inconclusive
dropped):
    n_test  ~= N_total / 10                  (one held-out fold / ~10% target)
    n_train ~= 2 * (0.8 * N_active)          (8 of 10 folds' actives, 1:1 downsampled)
This is an approximation (scaffold bins are not stratified, so true
per-repeat sizes vary a little), but it only scales the size of the
variance-inflation correction -- it does not affect the per-fold metric
values themselves, which are read directly from the existing results.

Comparisons tested per (assay, split-scheme), on ROC-AUC (threshold-free,
Test split):
  1. Best individual graph/GT model vs. best individual classical model
     -- the exact question Reviewer 1 raised (#11, #12): "is a 0.02 AUC
     difference statistically significant?"
  2. Consensus vs. best individual model -- the one comparison General
     Response G1(d) still claims to hold.
  3. MV-GNN-cross (Ma et al. baseline) vs. Consensus -- General Response
     G4(a).

Outputs (./results/):
  ci_summary.csv          -- mean, corrected SD, 95% CI, n, for every
                              model x metric x CPT (where applicable) x
                              assay x split-scheme.
  significance_tests.csv  -- the 3 comparisons above, both tests, and the
                              Holm-adjusted p-value within each family.

Usage: python compute_ci_significance.py
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.05

ASSAYS = {
    'AID_720637 (MMP summary)': {
        'wj': f'{BASE}/Summary_1_models/modeling/wahab_jobs',
        'cleaned_csv': f'{BASE}/Summary_1_models/cleaning/AID_720637_cleaned.csv',
    },
    'AID_654+1347034 (Caspase-3/7)': {
        'wj': f'{BASE}/summary_2_1437034/modeling/wahab_jobs',
        'cleaned_csv': f'{BASE}/summary_2_1437034/cleaning/AID_654_1347034_cleaned.csv',
    },
    'AID_1347037': {
        'wj': f'{BASE}/summary_1347037/modeling/wahab_jobs',
        'cleaned_csv': f'{BASE}/summary_1347037/cleaning/AID_1347037_cleaned.csv',
    },
}

MODEL_FILES = {
    'rf': 'Random Forest', 'svm': 'SVM', 'xgb': 'XGBoost',
    'gcn': 'GCN', 'gat': 'GAT', 'gin': 'GIN', 'sage': 'SAGE',
    'graphormer': 'Graphormer', 'graphgps': 'GraphGPS', 'mvgnn': 'MV-GNN-cross',
}
CLASSICAL_NAMES = {'Random Forest', 'SVM', 'XGBoost'}
GRAPH_NAMES = {'GCN', 'GAT', 'GIN', 'SAGE', 'Graphormer', 'GraphGPS', 'MV-GNN-cross'}
CONSENSUS_NAME = 'Consensus'

SPLIT_SCHEMES = {
    'LOO (10-fold)': {
        'dir': 'results_loo', 'suffix': 'loo_per_fold.csv', 'unit_col': 'Fold',
        'consensus_file': 'consensus_loo_per_fold.csv', 'n_folds_for_train_frac': 10,
    },
    'Scaffold (5-repeat)': {
        'dir': 'results_scaffold', 'suffix': 'scaffold_per_repeat.csv', 'unit_col': 'Repeat',
        'consensus_file': 'consensus_scaffold_per_repeat.csv', 'n_folds_for_train_frac': 10,
    },
}

THRESH_FREE_METRICS = ['ROC-AUC', 'PR-AUC']
THRESH_METRICS = ['Sensitivity', 'Specificity', 'CCR', 'MCC', 'Bal-Acc', 'F1']
PRIMARY_METRIC = 'ROC-AUC'


def get_dataset_counts(cleaned_csv):
    df = pd.read_csv(cleaned_csv, usecols=['PUBCHEM_ACTIVITY_OUTCOME'])
    counts = df['PUBCHEM_ACTIVITY_OUTCOME'].value_counts()
    n_active = int(counts.get('Active', 0))
    n_inactive = int(counts.get('Inactive', 0))
    return n_active + n_inactive, n_active, n_inactive


def approx_train_test_sizes(n_total, n_active):
    n_test = n_total / 10.0
    n_train = 2.0 * (0.8 * n_active)
    return n_train, n_test


def model_csv_path(wj, scheme_cfg, model_key):
    directory = os.path.join(wj, scheme_cfg['dir'])
    if model_key == 'consensus':
        return os.path.join(directory, scheme_cfg['consensus_file']), CONSENSUS_NAME
    return os.path.join(directory, f"{model_key}_{scheme_cfg['suffix']}"), MODEL_FILES[model_key]


def load_series(csv_path, model_name, unit_col, metric, cpt=None):
    df = pd.read_csv(csv_path)
    df = df[(df['Split'] == 'Test') & (df['Model'] == model_name)]
    if cpt is not None:
        df = df[df['CPT'] == cpt]
    else:
        df = df.drop_duplicates(subset=[unit_col])
    df = df.sort_values(unit_col)
    return df[unit_col].to_numpy(), df[metric].to_numpy(dtype=float)


def round_to_sig_figs(x, sig_figs):
    if x == 0:
        return 0.0, 0
    decimal_place = sig_figs - int(np.floor(np.log10(abs(x)))) - 1
    return round(x, decimal_place), decimal_place


def wheeler_precision(mean, moe):
    """Wheeler (2020) 'Level of Precision' rounding rule, applied to
    whichever mean/margin-of-error this script computes -- independent of
    which variance estimator (corrected or naive) produced the margin."""
    if moe == 0:
        return mean, moe
    leading_digit = int(f'{moe:.1e}'[0])
    sig_figs = 2 if leading_digit in (1, 2) else 1
    rounded_moe, decimal_place = round_to_sig_figs(moe, sig_figs)
    rounded_mean = round(mean, decimal_place)
    return rounded_mean, rounded_moe


def corrected_ci(values, n_train, n_test, alpha=ALPHA):
    n = len(values)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
    var = sd ** 2
    corrected_var_of_mean = (1.0 / n + n_test / n_train) * var
    se = float(np.sqrt(corrected_var_of_mean))
    if n > 1 and se > 0:
        tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        moe = tcrit * se
    else:
        moe = 0.0
    lo, hi = mean - moe, mean + moe
    mean_r, moe_r = wheeler_precision(mean, moe)
    return mean, sd, se, lo, hi, n, mean_r, moe_r


def corrected_paired_ttest(a, b, n_train, n_test):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1)) if n > 1 else 0.0
    corrected_var = (1.0 / n + n_test / n_train) * var_d
    se = float(np.sqrt(corrected_var))
    if se == 0:
        t_stat = 0.0 if mean_d == 0 else np.sign(mean_d) * np.inf
        p = 1.0 if mean_d == 0 else 0.0
        moe = 0.0
    else:
        t_stat = mean_d / se
        p = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        tcrit = stats.t.ppf(1 - ALPHA / 2, df=n - 1)
        moe = tcrit * se
    return mean_d, t_stat, p, moe


def wilcoxon_p(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(d, 0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(a, b)
        return float(stat), float(p)
    except ValueError:
        return np.nan, np.nan


def holm_bonferroni(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min((n - rank) * pvalues[idx], 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return adjusted


def main():
    ci_rows = []
    sig_rows = []

    for assay_label, cfg in ASSAYS.items():
        n_total, n_active, n_inactive = get_dataset_counts(cfg['cleaned_csv'])
        print(f'\n=== {assay_label}: N={n_total:,} (Active={n_active:,}, Inactive={n_inactive:,}) ===')

        for scheme_label, scheme_cfg in SPLIT_SCHEMES.items():
            n_train, n_test = approx_train_test_sizes(n_total, n_active)
            unit_col = scheme_cfg['unit_col']

            roc_auc_series = {}
            for model_key in list(MODEL_FILES) + ['consensus']:
                csv_path, model_name = model_csv_path(cfg['wj'], scheme_cfg, model_key)
                if not os.path.exists(csv_path):
                    print(f'  [skip] missing {csv_path}')
                    continue

                units, roc_vals = load_series(csv_path, model_name, unit_col, PRIMARY_METRIC)
                if len(roc_vals) == 0:
                    continue
                roc_auc_series[model_name] = (units, roc_vals)

                for metric in THRESH_FREE_METRICS:
                    _, vals = load_series(csv_path, model_name, unit_col, metric)
                    if len(vals) == 0:
                        continue
                    mean, sd, se, lo, hi, n, mean_r, moe_r = corrected_ci(vals, n_train, n_test)
                    ci_rows.append({'Assay': assay_label, 'SplitScheme': scheme_label, 'Model': model_name,
                                     'Metric': metric, 'CPT': None, 'N': n, 'Mean': round(mean, 4),
                                     'SD': round(sd, 4), 'Corrected_SE': round(se, 4),
                                     'CI95_Low': round(lo, 4), 'CI95_High': round(hi, 4),
                                     'Reported': f'{mean_r} +/- {moe_r}'})

                cpt_values = sorted(pd.read_csv(csv_path)['CPT'].unique())
                for cpt in cpt_values:
                    for metric in THRESH_METRICS:
                        _, vals = load_series(csv_path, model_name, unit_col, metric, cpt=cpt)
                        if len(vals) == 0:
                            continue
                        mean, sd, se, lo, hi, n, mean_r, moe_r = corrected_ci(vals, n_train, n_test)
                        ci_rows.append({'Assay': assay_label, 'SplitScheme': scheme_label, 'Model': model_name,
                                         'Metric': metric, 'CPT': cpt, 'N': n, 'Mean': round(mean, 4),
                                         'SD': round(sd, 4), 'Corrected_SE': round(se, 4),
                                         'CI95_Low': round(lo, 4), 'CI95_High': round(hi, 4),
                                         'Reported': f'{mean_r} +/- {moe_r}'})

            mean_roc = {m: np.mean(v[1]) for m, v in roc_auc_series.items()}
            classical_present = {m: v for m, v in mean_roc.items() if m in CLASSICAL_NAMES}
            graph_present = {m: v for m, v in mean_roc.items() if m in GRAPH_NAMES}
            if not classical_present or not graph_present or CONSENSUS_NAME not in roc_auc_series:
                print(f'  [skip comparisons] missing model group for {assay_label} / {scheme_label}')
                continue

            best_classical = max(classical_present, key=classical_present.get)
            best_graph = max(graph_present, key=graph_present.get)
            best_individual = max({**classical_present, **graph_present}, key={**classical_present, **graph_present}.get)

            comparisons = [
                ('Best graph/GT vs. best classical', best_graph, best_classical),
                ('Consensus vs. best individual', CONSENSUS_NAME, best_individual),
                ('MV-GNN-cross vs. Consensus', 'MV-GNN-cross', CONSENSUS_NAME),
            ]

            family_rows = []
            for comp_name, model_a, model_b in comparisons:
                if model_a not in roc_auc_series or model_b not in roc_auc_series:
                    continue
                units_a, a = roc_auc_series[model_a]
                units_b, b = roc_auc_series[model_b]
                common = np.intersect1d(units_a, units_b)
                a_aligned = a[np.isin(units_a, common)]
                b_aligned = b[np.isin(units_b, common)]
                mean_diff, t_stat, t_p, moe = corrected_paired_ttest(a_aligned, b_aligned, n_train, n_test)
                mean_r, moe_r = wheeler_precision(mean_diff, moe)
                w_stat, w_p = wilcoxon_p(a_aligned, b_aligned)
                family_rows.append({
                    'Assay': assay_label, 'SplitScheme': scheme_label, 'Comparison': comp_name,
                    'Model_A': model_a, 'Model_B': model_b, 'N': len(common),
                    'MeanROCAUC_A': round(float(np.mean(a_aligned)), 4),
                    'MeanROCAUC_B': round(float(np.mean(b_aligned)), 4),
                    'MeanDiff_A_minus_B': round(mean_diff, 4),
                    'Reported_Diff': f'{mean_r} +/- {moe_r}',
                    'CorrectedT_stat': round(t_stat, 4) if np.isfinite(t_stat) else t_stat,
                    'CorrectedT_pvalue': round(t_p, 4),
                    'Wilcoxon_stat': w_stat, 'Wilcoxon_pvalue': round(w_p, 4) if not np.isnan(w_p) else w_p,
                })

            if family_rows:
                fam_df = pd.DataFrame(family_rows)
                fam_df['CorrectedT_pvalue_Holm'] = holm_bonferroni(fam_df['CorrectedT_pvalue'].to_numpy()).round(4)
                fam_df['Wilcoxon_pvalue_Holm'] = holm_bonferroni(
                    fam_df['Wilcoxon_pvalue'].fillna(1.0).to_numpy()).round(4)
                fam_df['Significant_at_0.05_Holm'] = fam_df['CorrectedT_pvalue_Holm'] < ALPHA
                sig_rows.extend(fam_df.to_dict('records'))
                print(f'  {scheme_label}: n_train~={n_train:.0f}, n_test~={n_test:.0f}')
                for _, r in fam_df.iterrows():
                    print(f"    {r['Comparison']:35s} {r['Model_A']:16s} vs {r['Model_B']:16s}  "
                          f"diff={r['Reported_Diff']:>14s}  corrected-t p(Holm)={r['CorrectedT_pvalue_Holm']:.4f}  "
                          f"wilcoxon p(Holm)={r['Wilcoxon_pvalue_Holm']}")

    ci_df = pd.DataFrame(ci_rows)
    sig_df = pd.DataFrame(sig_rows)
    ci_df.to_csv(os.path.join(OUT_DIR, 'ci_summary.csv'), index=False)
    sig_df.to_csv(os.path.join(OUT_DIR, 'significance_tests.csv'), index=False)
    print(f'\nSaved -> {OUT_DIR}/ci_summary.csv ({len(ci_df)} rows)')
    print(f'Saved -> {OUT_DIR}/significance_tests.csv ({len(sig_df)} rows)')


if __name__ == '__main__':
    main()
