"""Paired significance tests for the UP-SAMPLED 10-fold runs.

compute_ci_significance.py covers only results_loo/ and results_scaffold/,
which are the 1:1 DOWNsampled runs. G5's "significant on MMP, p = 0.028"
therefore belongs to the downsampled result, not to the up-sampled runs the
tables now report. This recomputes the same comparisons on the up-sampled
per-fold files so the claim can be checked against the numbers actually
reported.

Method is transcribed from compute_ci_significance.py: Nadeau-Bengio corrected
resampled t-test, variance of the mean difference scaled by
(1/n + n_test/n_train) rather than 1/n because CV folds share training data,
plus a Wilcoxon signed-rank test, with Holm-Bonferroni correction inside each
assay/split family.

ONE DELIBERATE DIFFERENCE, AND WHY
----------------------------------
The correction needs n_train. For the downsampled runs the script uses
2 * (0.8 * n_active), the size of a 1:1 downsampled training set. Up-sampling
keeps every inactive and duplicates actives, so the training set is far larger
in rows but contains the same distinct compounds. Using the inflated row count
would shrink n_test/n_train and make the test anti-conservative. n_train is
therefore the number of DISTINCT training compounds, 0.8 * n_total, which is
the quantity the correction is about. Both variants are printed so the choice
is visible.

Usage: python compute_significance_upsampled.py
Output: results/significance_tests_upsampled.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05
CONSENSUS = {'Consensus', 'Consensus-Classical', 'Consensus-GNN',
             'Consensus-GT', 'Consensus-MVGNN'}

# assay -> (per-unit directory, filename pattern, n_total, n_active)
ASSAYS = {
    'AID_720637 (MMP summary)': (
        f'{HERE}/Summary_1_models/upsampling_protocols/results_loo',
        '{k}_per_unit.csv', 5937, 857),
    'AID_1347037 (Caspase CHO-K1)': (
        f'{HERE}/summary_1347037/upsampling/results_loo',
        '{k}_loo_per_fold.csv', 7496, 171),
}
KEYS = ['rf', 'svm', 'xgb', 'gcn', 'gat', 'gin', 'sage',
        'graphormer', 'graphgps', 'mvgnn', 'consensus']


def corrected_paired_ttest(a, b, n_train, n_test):
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return d.mean(), np.nan, np.nan
    se = np.sqrt((1.0 / n + n_test / n_train) * var)
    t = d.mean() / se
    return d.mean(), t, 2 * (1 - stats.t.cdf(abs(t), df=n - 1))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    out = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        out[idx] = min(run, 1.0)
    return out


def load(assay):
    d, pattern, _, _ = ASSAYS[assay]
    frames = []
    for k in KEYS:
        p = os.path.join(d, pattern.format(k=k))
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        t = t[(t.Split == 'Test') & (t.CPT == 0.5)]
        col = 'Fold' if 'Fold' in t.columns else 'Unit'
        frames.append(t[[col, 'Model', 'ROC-AUC']].rename(columns={col: 'Fold'}))
    return pd.concat(frames).drop_duplicates(['Fold', 'Model'])


def main():
    rows = []
    for assay, (_, _, n_total, n_active) in ASSAYS.items():
        df = load(assay)
        wide = df.pivot(index='Fold', columns='Model', values='ROC-AUC').dropna(axis=1)
        n_test = n_total / 10.0
        n_train_distinct = 0.8 * n_total          # distinct compounds in 8 folds
        n_train_rows = 2.0 * (0.8 * (n_total - n_active))   # up-sampled row count

        singles = [c for c in wide.columns if c not in CONSENSUS]
        best = wide[singles].mean().idxmax()
        print(f'\n=== {assay}   ({len(wide)} folds)')
        print(f'    best individual model: {best} ({wide[best].mean():.4f})')
        print(f'    Consensus:             {wide["Consensus"].mean():.4f}')

        for label, a, b in [('Consensus vs. best individual', 'Consensus', best)]:
            for tag, ntr in [('distinct', n_train_distinct), ('rows', n_train_rows)]:
                diff, t, p = corrected_paired_ttest(wide[a], wide[b], ntr, n_test)
                w, wp = stats.wilcoxon(wide[a], wide[b])
                rows.append({'Assay': assay, 'SplitScheme': 'LOO (10-fold), up-sampled',
                             'Comparison': label, 'Model_A': a, 'Model_B': b,
                             'n_train_basis': tag, 'N': len(wide),
                             'MeanROCAUC_A': round(wide[a].mean(), 4),
                             'MeanROCAUC_B': round(wide[b].mean(), 4),
                             'MeanDiff': round(diff, 4),
                             'CorrectedT_stat': round(t, 4),
                             'CorrectedT_pvalue': round(p, 4),
                             'Wilcoxon_pvalue': round(wp, 4)})
                print(f'    n_train({tag})={ntr:.0f}: diff={diff:+.4f} '
                      f'corrected-t p={p:.4f}   Wilcoxon p={wp:.4f}')

    out = pd.DataFrame(rows)
    for tag in ('distinct', 'rows'):
        m = out.n_train_basis == tag
        out.loc[m, 'CorrectedT_pvalue_Holm'] = holm(out.loc[m, 'CorrectedT_pvalue']).round(4)
        out.loc[m, 'Wilcoxon_pvalue_Holm'] = holm(out.loc[m, 'Wilcoxon_pvalue']).round(4)
    out['Significant_at_0.05_Holm'] = out.CorrectedT_pvalue_Holm <= ALPHA
    p = f'{HERE}/results/significance_tests_upsampled.csv'
    out.to_csv(p, index=False)
    print(f'\nwrote {p}')


if __name__ == '__main__':
    main()
