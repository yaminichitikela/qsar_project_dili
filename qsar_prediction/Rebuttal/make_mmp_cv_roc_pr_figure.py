"""ROC and PR curves for the MMP summary assay (AID 720637), 10-fold CV with
random up-sampling (1:1) of the training set -- the run behind tab:mmp_up_loo.
Top 5 models by mean test ROC-AUC. Companion to
make_chok1_cv_roc_pr_figures.py; same layout and colours.

DATA
----
Summary_1_models/upsampling_protocols: per-fold test probabilities
(results_loo/probs), test labels (data_loo/fold_<i>/y_test.npy) and summary
tables (results_loo/*_summary.csv). The transformers there are already the
corrected architecture. Consensus probabilities were not saved, so they are
rebuilt with the groupings of compute_consensus.py (unweighted mean; MV-GNN is
excluded from the full consensus, as in the paper).

Each plotted curve is the vertical average of the 10 fold curves; legend values
are the mean per-fold test AUCs from the summary tables (CPT 0.5), i.e. the
values in tab:mmp_up_loo. The script recomputes every model's mean ROC-AUC and
stops if it disagrees with the table.

Usage: python make_mmp_cv_roc_pr_figure.py
Outputs: figures/mmp_cv_upsampling_roc_pr.png/.pdf
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__))
UPP = os.path.join(HERE, 'Summary_1_models', 'upsampling_protocols')
RES = os.path.join(UPP, 'results_loo')
OUT = os.path.join(HERE, 'figures')
N_FOLDS = 10

GNN = ['gcn', 'gat', 'gin', 'sage']
GT = ['graphormer', 'graphgps']
ALL9 = ['rf', 'svm', 'xgb'] + GNN + GT

# name -> (member models averaged, summary table, colour, linestyle, linewidth)
# Colour = model family; solid = consensus, dashed/dash-dot = single model.
MODELS = {
    'Consensus':     (ALL9,    'consensus_summary.csv', '#3f3e3b', '-',  2.6),
    'Consensus-GNN': (GNN,     'consensus_summary.csv', '#1baf7a', '-',  2.2),
    'GIN':           (['gin'], 'gin_summary.csv',       '#1baf7a', '--', 1.6),
    'Consensus-GT':  (GT,      'consensus_summary.csv', '#eb6834', '-',  2.2),
    'GCN':           (['gcn'], 'gcn_summary.csv',       '#1baf7a', '-.', 1.6),
}


def table_value(name, metric):
    df = pd.read_csv(os.path.join(RES, MODELS[name][1]))
    row = df[(df['Model'] == name) & (df['CPT'] == 0.5) & (df['Stat'] == 'mean')]
    return float(row[metric].iloc[0])


def main():
    labels = [np.load(os.path.join(UPP, 'data_loo', f'fold_{f}', 'y_test.npy')).astype(int)
              for f in range(N_FOLDS)]
    probs = [{m: np.load(os.path.join(RES, 'probs', f'{m}_fold{f}_test_probs.npy')) for m in ALL9}
             for f in range(N_FOLDS)]
    y_all = np.concatenate(labels)
    print(f'Compounds: {len(y_all):,}  active={y_all.sum():,}')

    grid = np.linspace(0, 1, 501)
    curves = {}
    for name, (members, *_ ) in MODELS.items():
        tprs, precs, aucs = [], [], []
        for f in range(N_FOLDS):
            y = labels[f]
            p = np.mean([probs[f][m] for m in members], axis=0)
            fpr, tpr, _ = roc_curve(y, p)
            t = np.interp(grid, fpr, tpr); t[0] = 0.0
            tprs.append(t)
            pre, rec, _ = precision_recall_curve(y, p)
            precs.append(np.interp(grid, rec[::-1], np.maximum.accumulate(pre)[::-1]))
            aucs.append(roc_auc_score(y, p))
        ref = table_value(name, 'ROC-AUC')
        if abs(np.mean(aucs) - ref) > 1e-3:
            raise SystemExit(f'{name}: recomputed mean ROC-AUC {np.mean(aucs):.4f} '
                             f'!= table {ref:.4f} -- not plotting')
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        curves[name] = dict(tpr=mean_tpr, prec=np.mean(precs, axis=0))
        print(f'{name:14s} ROC-AUC {np.mean(aucs):.4f} (table {ref:.4f})')

    fmt = lambda v: f'{round(v + 1e-9, 3):.3f}'   # half-up, matches the tables
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#52514e', 'axes.linewidth': 0.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    for name, (_, _, color, ls, lw) in MODELS.items():
        ax1.plot(grid, curves[name]['tpr'], color=color, ls=ls, lw=lw,
                 label=f'{name} ({fmt(table_value(name, "ROC-AUC"))})')
        ax2.plot(grid, curves[name]['prec'], color=color, ls=ls, lw=lw,
                 label=f'{name} ({fmt(table_value(name, "PR-AUC"))})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='FPR', ylabel='TPR', title='ROC Curve')
    ax2.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Recall', ylabel='Precision', title='PR Curve')
    ax1.legend(loc='lower right', fontsize=10, title='Mean ROC-AUC', title_fontsize=10)
    ax2.legend(loc='lower left', fontsize=10, title='Mean PR-AUC', title_fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'mmp_cv_upsampling_roc_pr.{ext}'), dpi=300)
    plt.close(fig)
    print(f'Saved {os.path.join(OUT, "mmp_cv_upsampling_roc_pr.png")}')


if __name__ == '__main__':
    main()
