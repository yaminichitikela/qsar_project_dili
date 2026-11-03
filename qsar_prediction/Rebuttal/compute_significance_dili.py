"""Paired significance tests for the DILI held-out test set (Held-223, 222 compounds).

Reviewer 1, Comment 3. Same method as compute_ci_significance.py /
compute_significance_upsampled.py: Nadeau--Bengio corrected resampled t-test
(variance of the mean difference scaled by 1/n + n_test/n_train, because the
units share training data), Wilcoxon signed-rank as a check, Holm correction
within each protocol x stage family.

Reported DILI setting: 1:1 down-sampled training, cutoff 0.5, three protocols
(5 random repeats, 10-fold CV, 5 scaffold repeats), Proxy-only and Proxy+DILI.
The full consensus is compared with each of the nine base models, MV-GNN-cross
and the replicated Xu et al. structure-only model (2,048 bits, top 360).
n_train follows make_assay_style_tables.py: mean idx_train size, plus NCV-888
in stage 2.

DILIPredictor (Seal et al., ROC-AUC 0.63) has no per-compound predictions, so
it cannot be paired; it is checked against the consensus 95% interval instead.

Usage: python compute_significance_dili.py
Output: results/significance_tests_dili.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DSP = f'{HERE}/DILI_split_protocols'
XU = f'{HERE}/DILI_tx2017_2048/results'
PROTOCOLS = {'random_repeat': 5, 'cv': 10, 'scaffold': 5}
METRICS = ['ROC-AUC', 'PR-AUC', 'F1']
BASE = ['RF', 'SVM', 'XGB', 'GCN', 'GAT', 'GIN', 'SAGE', 'Graphormer', 'GraphGPS', 'MVGNN', 'Structure only']
N_TEST = 222
DILIPREDICTOR_AUC = 0.63


def n_train(proto, stage):
    n_ncv = len(np.load(f'{DSP}/data/y_ncv.npy'))
    tr = [len(np.load(f'{DSP}/splits/{proto}/unit_{u}/idx_train.npy')) for u in range(PROTOCOLS[proto])]
    return float(np.mean(tr)) + (n_ncv if stage == 2 else 0)


def nb_test(diff, ntr):
    n = len(diff)
    var = diff.var(ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = diff.mean() / np.sqrt((1 / n + N_TEST / ntr) * var)
    return t, 2 * stats.t.sf(abs(t), df=n - 1)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    per = pd.read_csv(f'{DSP}/results/per_unit_all.csv')
    per = per[(per.Split == 'test') & (per.CPT == 0.5)]
    rows, ref = [], []
    for proto in PROTOCOLS:
        xu = pd.read_csv(f'{XU}/arm_n360_{proto}.csv')
        xu = xu[xu.CPT == 0.5].assign(Model='Structure only')[['Stage', 'Unit', 'Model'] + METRICS]
        d = pd.concat([per[per.Protocol == proto][['Stage', 'Unit', 'Model'] + METRICS], xu])
        for stage in (1, 2):
            s = d[d.Stage == stage]
            ntr = n_train(proto, stage)
            cons = s[s.Model == 'Consensus'].set_index('Unit').sort_index()
            assert len(cons) == PROTOCOLS[proto], (proto, stage, len(cons))
            # consensus vs DILIPredictor's published ROC-AUC (one-sample, NB-corrected)
            v = cons['ROC-AUC'].values
            t, p = nb_test(v - DILIPREDICTOR_AUC, ntr)
            half = stats.t.ppf(0.975, len(v) - 1) * np.sqrt((1 / len(v) + N_TEST / ntr) * v.var(ddof=1))
            ref.append(dict(Protocol=proto, Stage=stage, Consensus_ROC_AUC=v.mean(), CI_low=v.mean() - half,
                            CI_high=v.mean() + half, DILIPredictor=DILIPREDICTOR_AUC, p_value=p))
            for metric in METRICS:
                fam = []
                for m in BASE:
                    b = s[s.Model == m].set_index('Unit').sort_index()
                    assert list(b.index) == list(cons.index), (proto, stage, m)
                    diff = cons[metric].values - b[metric].values
                    t, p = nb_test(diff, ntr)
                    try:
                        pw = stats.wilcoxon(diff).pvalue
                    except ValueError:
                        pw = np.nan
                    fam.append(dict(Protocol=proto, Stage=stage, Metric=metric, Model_A='Consensus', Model_B=m,
                                    N=len(diff), Mean_A=cons[metric].mean(), Mean_B=b[metric].mean(),
                                    MeanDiff=diff.mean(), NB_t=t, NB_p=p, Wilcoxon_p=pw))
                adj = holm([r['NB_p'] for r in fam])
                for r, a in zip(fam, adj):
                    r['NB_p_Holm'] = a
                    r['Significant_0.05_Holm'] = bool(a < 0.05)
                rows += fam
    out = pd.DataFrame(rows)
    out.to_csv(f'{HERE}/results/significance_tests_dili.csv', index=False)
    pd.DataFrame(ref).to_csv(f'{HERE}/results/significance_dili_vs_dilipredictor.csv', index=False)

    pd.set_option('display.width', 220)
    print('Consensus vs DILIPredictor (ROC-AUC 0.63):')
    print(pd.DataFrame(ref).round(4).to_string(index=False))
    a = out[out.Metric == 'ROC-AUC']
    print('\nROC-AUC, consensus vs each model (NB-corrected, Holm within protocol x stage):')
    print(a[['Protocol', 'Stage', 'Model_B', 'Mean_A', 'Mean_B', 'MeanDiff', 'NB_p', 'NB_p_Holm',
             'Significant_0.05_Holm']].round(4).to_string(index=False))
    print('\nAll significant (any metric):')
    sig = out[out['Significant_0.05_Holm']]
    print(sig[['Protocol', 'Stage', 'Metric', 'Model_B', 'Mean_A', 'Mean_B', 'NB_p_Holm']].round(4).to_string(index=False)
          if len(sig) else '  none')


if __name__ == '__main__':
    main()
