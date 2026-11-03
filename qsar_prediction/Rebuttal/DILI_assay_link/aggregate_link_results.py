"""Turns the per-repeat linking runs into the answer, with uncertainty.

THE QUESTION
------------
Does adding the two in vitro HepG2 predictions to a DILI model improve it?
Two arms, identical except their inputs, five repeats each, scored on the fixed
Held-223 test set:

    structure       ECFP4 / graphs only
    structure_iv    the same, plus P(MMP active) and P(caspase-3/7 active)

WHAT THIS COMPUTES
------------------
 1. mean +/- SD across repeats, per arm / family / stage
 2. the consensus ensembles, built per repeat from the member families
 3. a bootstrap 95% CI on Held-223 for the consensus of each arm, using the same
    2000-resample percentile method as table_helpers.bootstrap_ci_metric, so the
    numbers are comparable with every other table in the rebuttal
 4. a paired comparison between arms: the two arms share repeats and share the
    test set, so the per-repeat differences are paired, and a paired t-test plus
    a Wilcoxon signed-rank test are reported on them

A difference is only called real if the paired test supports it. With 5 repeats
the test is weak, which is itself worth stating plainly rather than reading a
sign off two decimal places.

Usage: python aggregate_link_results.py
Outputs (./results/link/):
    per_repeat_all.csv        -- every arm/family/stage/repeat row
    linking_summary.csv       -- mean +/- SD per arm/family/stage
    linking_comparison.csv    -- the arm-vs-arm paired tests  <- the headline
"""
import glob
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))      # for table_helpers
from eval_utils import METRIC_COLS, evaluate

ARMS = ['structure', 'structure_iv']
STAGES = [1, 2]
CLASSICAL = ['RF', 'SVM', 'XGB']
GNN = ['GCN', 'GAT', 'GIN', 'SAGE']
GT = ['Graphormer', 'GraphGPS']
MVGNN = 'MVGNN'
N_BOOT, BOOT_SEED = 2000, 42
METRICS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC',
           'Bal-Acc', 'F1']
DATA = os.path.join(HERE, 'data_dili_link')
OUT = os.path.join(HERE, 'results', 'link')


def bootstrap_ci(y, p, metric='ROC-AUC', n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap, matching table_helpers.bootstrap_ci_metric."""
    fn = roc_auc_score if metric == 'ROC-AUC' else average_precision_score
    rng = np.random.RandomState(seed)
    n, vals = len(y), []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return max(lo, 0.0), min(hi, 1.0)


def load_per_repeat():
    rows = []
    for arm in ARMS:
        for f in sorted(glob.glob(f'{OUT}/{arm}/*_s*_r*_metrics.csv')):
            rows.append(pd.read_csv(f))
    if not rows:
        print(f'No per-repeat metrics under {OUT} -- run the fits first')
        sys.exit(1)
    return pd.concat(rows, ignore_index=True)


def consensus_rows(y):
    """Builds the consensus ensembles per (arm, stage, repeat) from the member
    families' saved Held-223 probabilities, and returns both the metric rows and
    the probability vectors (needed for the bootstrap CI)."""
    rows, probs_out = [], {}
    for arm in ARMS:
        for stage in STAGES:
            for r in range(99):
                got = {}
                for fam in CLASSICAL + GNN + GT + [MVGNN]:
                    p = f'{OUT}/{arm}/probs/{fam}_s{stage}_r{r}_test.npy'
                    if os.path.exists(p):
                        got[fam] = np.load(p)
                if not got:
                    break
                nine = [f for f in CLASSICAL + GNN + GT if f in got]
                ens = {}
                if all(f in got for f in CLASSICAL):
                    ens['Consensus-Classical'] = np.mean([got[f] for f in CLASSICAL], axis=0)
                if all(f in got for f in GNN):
                    ens['Consensus-GNN'] = np.mean([got[f] for f in GNN], axis=0)
                if all(f in got for f in GT):
                    ens['Consensus-GT'] = np.mean([got[f] for f in GT], axis=0)
                if len(nine) == 9:
                    ens['Consensus'] = np.mean([got[f] for f in nine], axis=0)
                    if MVGNN in got:
                        ens['Consensus-MVGNN'] = np.mean(
                            [got[f] for f in nine + [MVGNN]], axis=0)
                for name, pr in ens.items():
                    probs_out[(arm, stage, r, name)] = pr
                    for cpt in (0.5, 0.8):
                        rows.append({**evaluate(y, pr, name, cpt), 'Arm': arm,
                                     'Stage': stage, 'Repeat': r, 'Split': 'test'})
    return pd.DataFrame(rows), probs_out


def main():
    y = np.load(f'{DATA}/y_test.npy').astype(int)
    print(f'Held-223 test set: {len(y)} compounds, DILI+ {100 * y.mean():.1f}%')

    per = load_per_repeat()
    per.columns = [c.strip() for c in per.columns]
    cons, cons_probs = consensus_rows(y)
    allrows = pd.concat([per, cons], ignore_index=True) if len(cons) else per
    test = allrows[allrows.Split.str.lower() == 'test'].copy()
    os.makedirs(OUT, exist_ok=True)
    test.to_csv(f'{OUT}/per_repeat_all.csv', index=False)

    # ── 1-2. mean +/- SD per arm / family / stage ───────────────────────────
    summ = []
    for (arm, stage, model, cpt), g in test.groupby(['Arm', 'Stage', 'Model', 'CPT']):
        row = {'Arm': arm, 'Stage': stage, 'Model': model, 'CPT': cpt,
               'n_repeats': g.Repeat.nunique()}
        for m in METRICS:
            if m in g:
                row[f'{m}_mean'] = round(float(g[m].mean()), 4)
                row[f'{m}_sd'] = round(float(g[m].std(ddof=1)) if len(g) > 1 else 0.0, 4)
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(f'{OUT}/linking_summary.csv', index=False)

    # ── 3-4. arm vs arm, paired over repeats ────────────────────────────────
    comp = []
    for stage in STAGES:
        for model in sorted(test.Model.unique()):
            a = test[(test.Arm == 'structure') & (test.Stage == stage)
                     & (test.Model == model) & (test.CPT == 0.5)].set_index('Repeat')
            b = test[(test.Arm == 'structure_iv') & (test.Stage == stage)
                     & (test.Model == model) & (test.CPT == 0.5)].set_index('Repeat')
            shared = sorted(set(a.index) & set(b.index))
            if len(shared) < 2:
                continue
            x, z = a.loc[shared, 'ROC-AUC'].values, b.loc[shared, 'ROC-AUC'].values
            d = z - x
            t_p = stats.ttest_rel(z, x).pvalue if len(shared) > 1 and np.std(d) > 0 else 1.0
            try:
                w_p = stats.wilcoxon(z, x).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                w_p = 1.0
            row = dict(stage=stage, model=model, n_repeats=len(shared),
                       structure_auc=round(float(x.mean()), 4),
                       structure_iv_auc=round(float(z.mean()), 4),
                       delta=round(float(d.mean()), 4),
                       delta_sd=round(float(np.std(d, ddof=1)) if len(d) > 1 else 0, 4),
                       paired_t_p=round(float(t_p), 4),
                       wilcoxon_p=round(float(w_p), 4))
            # bootstrap CI on the consensus, from repeat 0's probabilities
            if model == 'Consensus':
                for arm, tag in (('structure', 'structure'), ('structure_iv', 'structure_iv')):
                    pr = cons_probs.get((arm, stage, 0, 'Consensus'))
                    if pr is not None:
                        lo, hi = bootstrap_ci(y, pr)
                        row[f'{tag}_ci95'] = f'{lo:.3f}-{hi:.3f}'
            comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(f'{OUT}/linking_comparison.csv', index=False)

    # ── report ──────────────────────────────────────────────────────────────
    print(f'\n{len(test)} test rows over {test.Repeat.nunique()} repeats, '
          f'{test.Model.nunique()} models, arms {sorted(test.Arm.unique())}')
    print('\n' + '=' * 86)
    print('DOES THE IN VITRO INFORMATION HELP?  (Held-223 ROC-AUC, cutoff 0.5)')
    print('=' * 86)
    if len(comp):
        show = comp[['stage', 'model', 'n_repeats', 'structure_auc',
                     'structure_iv_auc', 'delta', 'paired_t_p', 'wilcoxon_p']]
        print(show.to_string(index=False))
        sig = comp[(comp.paired_t_p < 0.05)]
        print(f'\n{len(sig)} of {len(comp)} comparisons reach p < 0.05 (paired t-test)')
        if len(sig):
            print(sig[['stage', 'model', 'delta', 'paired_t_p']].to_string(index=False))
        else:
            print('None. With 5 repeats the test is weak, so this is evidence of')
            print('no detectable effect rather than proof of no effect.')
    print(f'\n-> {OUT}/linking_summary.csv')
    print(f'-> {OUT}/linking_comparison.csv   <- headline')


if __name__ == '__main__':
    main()
