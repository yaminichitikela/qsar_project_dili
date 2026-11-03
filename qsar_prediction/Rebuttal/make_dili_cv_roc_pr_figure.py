"""ROC and PR curves on the DILI Held-223 test set (222 compounds, 155
hepatotoxic) for the 10-fold CV, 1:1 downsampled protocol behind
tab:dili_cv_downsampled. Top 5 models per stage by mean ROC-AUC; Stage 1
(Proxy-only) on the top row, Stage 2 (Proxy+DILI) on the bottom row.
Same colours and line styles as the CHO-K1 and MMP CV figures.

DATA
----
DILI_split_protocols: per-fold Held-223 probabilities
(results/cv/probs/<Model>_s<stage>_u<fold>_test.npy), labels (data/y_test.npy)
and the summary table (results/summary.csv). Consensus probabilities are the
unweighted mean of their members (Classical = RF, SVM, XGB; GNN = GCN, GAT,
GIN, SAGE; GT = Graphormer, GraphGPS; full Consensus = all 9, MV-GNN excluded).

Every fold scores the same 222 compounds, so each plotted curve is the vertical
average of the 10 fold curves. Legend values are the table means (CPT 0.5); the
script recomputes each model's mean ROC-AUC and stops if it disagrees.

Usage: python make_dili_cv_roc_pr_figure.py          # Stage 2 only (main paper)
       python make_dili_cv_roc_pr_figure.py both     # Stage 1 and Stage 2
Outputs: figures/dili_cv_downsampled_stage2_roc_pr.png/.pdf
         figures/dili_cv_downsampled_roc_pr.png/.pdf (with 'both')
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
DSP = os.path.join(HERE, 'DILI_split_protocols')
OUT = os.path.join(HERE, 'figures')
N_FOLDS = 10

GROUPS = {
    'Consensus': ['RF', 'SVM', 'XGB', 'GCN', 'GAT', 'GIN', 'SAGE', 'Graphormer', 'GraphGPS'],
    'Consensus-GNN': ['GCN', 'GAT', 'GIN', 'SAGE'],
    'Consensus-GT': ['Graphormer', 'GraphGPS'],
}
# display name -> (summary.csv name, colour, linestyle, linewidth)
# Colour = model family; solid = consensus, dashed = single model.
STYLE = {
    'Consensus':     ('Consensus',     '#3f3e3b', '-',  2.6),
    'Consensus-GNN': ('Consensus-GNN', '#1baf7a', '-',  2.2),
    'Consensus-GT':  ('Consensus-GT',  '#eb6834', '-',  2.2),
    'GIN':           ('GIN',           '#1baf7a', '--', 1.6),
    'Graphormer':    ('Graphormer',    '#eb6834', '--', 1.6),
    'XGBoost':       ('XGB',           '#2a78d6', '--', 1.6),
}
TOP5 = {  # ranked by mean ROC-AUC in tab:dili_cv_downsampled, MV-GNN excluded
    1: ['Consensus-GT', 'Consensus', 'Graphormer', 'Consensus-GNN', 'GIN'],
    2: ['Consensus', 'Consensus-GNN', 'GIN', 'Consensus-GT', 'XGBoost'],
}
STAGE_TITLE = {1: 'Stage 1 (Proxy-only)', 2: 'Stage 2 (Proxy+DILI)'}

summary = pd.read_csv(os.path.join(DSP, 'results', 'summary.csv'))
summary = summary[(summary['Protocol'] == 'cv') & (summary['Split'] == 'test')
                  & (summary['CPT'] == 0.5)]


def table_value(stage, name, metric):
    row = summary[(summary['Stage'] == stage) & (summary['Model'] == STYLE[name][0])]
    return float(row[f'{metric}_mean'].iloc[0])


def fold_probs(stage, name, fold):
    key = STYLE[name][0]
    members = GROUPS.get(key, [key])
    return np.mean([np.load(os.path.join(DSP, 'results', 'cv', 'probs',
                                         f'{m}_s{stage}_u{fold}_test.npy'))
                    for m in members], axis=0)


def main():
    y = np.load(os.path.join(DSP, 'data', 'y_test.npy')).astype(int)
    print(f'Held-223: {len(y)} compounds, {y.sum()} hepatotoxic')
    grid = np.linspace(0, 1, 501)
    fmt = lambda v: f'{round(v + 1e-9, 3):.3f}'   # half-up, matches the tables

    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#52514e', 'axes.linewidth': 0.8})
    stages = (1, 2) if 'both' in sys.argv[1:] else (2,)
    fig, axes = plt.subplots(len(stages), 2, figsize=(13, 5.6 * len(stages)), squeeze=False)
    for r, stage in enumerate(stages):
        ax1, ax2 = axes[r]
        for name in TOP5[stage]:
            tprs, precs, aucs = [], [], []
            for f in range(N_FOLDS):
                p = fold_probs(stage, name, f)
                fpr, tpr, _ = roc_curve(y, p)
                t = np.interp(grid, fpr, tpr); t[0] = 0.0
                tprs.append(t)
                pre, rec, _ = precision_recall_curve(y, p)
                precs.append(np.interp(grid, rec[::-1], np.maximum.accumulate(pre)[::-1]))
                aucs.append(roc_auc_score(y, p))
            ref = table_value(stage, name, 'ROC-AUC')
            if abs(np.mean(aucs) - ref) > 1e-3:
                raise SystemExit(f'Stage {stage} {name}: recomputed {np.mean(aucs):.4f} '
                                 f'!= table {ref:.4f} -- not plotting')
            print(f'Stage {stage} {name:14s} ROC-AUC {np.mean(aucs):.4f} (table {ref:.4f})')
            mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
            _, color, ls, lw = STYLE[name]
            ax1.plot(grid, mean_tpr, color=color, ls=ls, lw=lw,
                     label=f'{name} ({fmt(ref)})')
            ax2.plot(grid, np.mean(precs, axis=0), color=color, ls=ls, lw=lw,
                     label=f'{name} ({fmt(table_value(stage, name, "PR-AUC"))})')
        ax1.plot([0, 1], [0, 1], 'k--', lw=1)
        ax1.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='FPR', ylabel='TPR',
                title=f'ROC Curve — {STAGE_TITLE[stage]}')
        ax2.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Recall', ylabel='Precision',
                title=f'PR Curve — {STAGE_TITLE[stage]}')
        ax1.legend(loc='lower right', fontsize=10, title='Mean ROC-AUC', title_fontsize=10)
        ax2.legend(loc='lower left', fontsize=10, title='Mean PR-AUC', title_fontsize=10)
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    stem = 'dili_cv_downsampled_roc_pr' if len(stages) == 2 else 'dili_cv_downsampled_stage2_roc_pr'
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{stem}.{ext}'), dpi=300)
    plt.close(fig)
    print(f'Saved {os.path.join(OUT, stem + ".png")}')


if __name__ == '__main__':
    main()
