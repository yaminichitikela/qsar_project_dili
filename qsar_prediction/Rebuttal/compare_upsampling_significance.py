"""
Paired DeLong test comparing 1:1-downsampling ROC-AUC against upsampling
ROC-AUC, model by model, for each dataset. Valid here specifically because
both runs share the exact same held-out test set (verified byte-identical
against the downsampling run's own y_test/X_test during setup) -- only the
training data differs. This is a different (and here, more appropriate)
test than the Nadeau-Bengio approach used elsewhere in the letter, which
needs repeated splits; this experiment used a single 80/10/10 split, so a
paired test on the shared test set is the valid option, not a repeated-CV
variance estimate.

DeLong's method (Sun & Xu, 2014 fast version) via the placement-values
formulation, giving both AUCs, their difference, its variance, and a
two-sided p-value from the z-statistic.

Usage: python compare_upsampling_significance.py
"""
import glob
import os

import numpy as np
from scipy import stats


def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / (2 * n)

    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, prob_a, prob_b):
    """Two-sided paired DeLong test between two sets of scores on the same
    y_true. Returns (auc_a, auc_b, auc_diff, p_value)."""
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind='stable')
    y_sorted = y_true[order]
    m = int(y_sorted.sum())

    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, cov = fastDeLong(preds, m)
    auc_diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], auc_diff, 1.0
    z = auc_diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], auc_diff, p


DATASETS = [
    ('MMP', 'Summary_1_models/modeling/wahab_jobs/results', 'Summary_1_models/upsampling/results',
     'Summary_1_models/upsampling/data/y_test.npy'),
    ('Caspase HepG2', 'summary_2_1437034/modeling/wahab_jobs/results', 'summary_2_1437034/upsampling/results',
     'summary_2_1437034/upsampling/data/y_test.npy'),
    ('Caspase CHO-K1', 'summary_1347037/modeling/wahab_jobs/results', 'summary_1347037/upsampling/results',
     'summary_1347037/upsampling/data/y_test.npy'),
]

HERE = os.path.dirname(os.path.abspath(__file__))

for name, down_dir, up_dir, y_test_path in DATASETS:
    y_test = np.load(os.path.join(HERE, y_test_path))
    print(f'=== {name} (n_test={len(y_test)}) ===')
    down_models = {os.path.basename(f).replace('_test_probs.npy', '')
                   for f in glob.glob(os.path.join(HERE, down_dir, '*_test_probs.npy'))}
    up_models = {os.path.basename(f).replace('_test_probs.npy', '')
                 for f in glob.glob(os.path.join(HERE, up_dir, '*_test_probs.npy'))}
    common = sorted(down_models & up_models)
    for m in common:
        prob_down = np.load(os.path.join(HERE, down_dir, f'{m}_test_probs.npy'))
        prob_up = np.load(os.path.join(HERE, up_dir, f'{m}_test_probs.npy'))
        if len(prob_down) != len(y_test) or len(prob_up) != len(y_test):
            print(f'  {m:10s} SKIPPED -- length mismatch (down={len(prob_down)}, up={len(prob_up)}, y={len(y_test)})')
            continue
        auc_down, auc_up, diff, p = delong_roc_test(y_test, prob_down, prob_up)
        sig = '*' if p < 0.05 else ' '
        print(f'  {m:10s}  1:1={auc_down:.4f}  up={auc_up:.4f}  diff={diff:+.4f}  p={p:.4f} {sig}')
    print()
