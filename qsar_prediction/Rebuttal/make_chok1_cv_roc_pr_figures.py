"""ROC and PR curves for CHO-K1 Caspase-3/7 (AID 1347037), 10-fold CV with
up-sampling, top 5 models by mean test ROC-AUC.

DATA
----
Graph transformer and consensus probabilities come from the corrected runs
(graph_transformer_corrected/results/chok1/up_loo/probs), whose summary matches
the numbers in the paper (Consensus 0.837, Consensus-GT 0.823, Consensus-GNN
0.812). GNN probabilities come from summary_1347037/upsampling/results_loo/probs.

The per-fold test labels (data_loo/) were not kept locally, so they are rebuilt
with the exact fold assignment of prepare_loo_folds_upsample.py: same cleaning,
same StratifiedKFold(10, shuffle=True, random_state=42). The script recomputes
every fold's ROC-AUC from the rebuilt labels and refuses to plot unless it
matches the saved per-fold metrics.

CURVES
------
Each fold gives its own curve. The plotted curve is the vertical average over
the 10 folds (TPR interpolated on a common FPR grid; precision interpolated on a
common recall grid). Legend values are read from the CV summary tables
(mean of per-fold test AUCs at CPT 0.5), so they are exactly the numbers
reported in the paper.

Usage: python make_chok1_cv_roc_pr_figures.py
Outputs: figures/chok1_cv_upsampling_roc_pr.png/.pdf
"""
import os
import warnings; warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.join(HERE, 'summary_1347037', 'upsampling')
GT_CORR = os.path.join(HERE, 'graph_transformer_corrected', 'results', 'chok1', 'up_loo')
OUT = os.path.join(HERE, 'figures')
N_FOLDS, SEED = 10, 42

# name -> (probability file pattern, colour, linestyle, linewidth)
# Colour = model family (same palette as make_roc_pr_figure.py);
# solid = consensus, dashed = single model.
MODELS = {
    'Consensus':     (os.path.join(GT_CORR, 'probs', 'Consensus_{f}_test_probs.npy'),     '#3f3e3b', '-',  2.6),
    'Consensus-GT':  (os.path.join(GT_CORR, 'probs', 'Consensus-GT_{f}_test_probs.npy'),  '#eb6834', '-',  2.2),
    'Consensus-GNN': (os.path.join(GT_CORR, 'probs', 'Consensus-GNN_{f}_test_probs.npy'), '#1baf7a', '-',  2.2),
    'GIN':           (os.path.join(UP, 'results_loo', 'probs', 'gin_fold{f}_test_probs.npy'), '#1baf7a', '--', 1.6),
    'GraphGPS':      (os.path.join(GT_CORR, 'probs', 'GraphGPS_{f}_test_probs.npy'),      '#eb6834', '--', 1.6),
}


SUMMARY = {
    'Consensus':     (os.path.join(GT_CORR, 'consensus_summary.csv'), 'Consensus'),
    'Consensus-GT':  (os.path.join(GT_CORR, 'consensus_summary.csv'), 'Consensus-GT'),
    'Consensus-GNN': (os.path.join(GT_CORR, 'consensus_summary.csv'), 'Consensus-GNN'),
    'GIN':           (os.path.join(UP, 'results_loo', 'gin_loo_summary.csv'), 'GIN'),
    'GraphGPS':      (os.path.join(GT_CORR, 'GraphGPS_summary.csv'), 'GraphGPS'),
}


def table_value(name, metric):
    path, model = SUMMARY[name]
    df = pd.read_csv(path)
    row = df[(df['Model'] == model) & (df['CPT'] == 0.5) & (df['Stat'] == 'mean')]
    # half-up rounding, so 0.8115 prints as 0.812 like the paper tables
    return round(float(row[metric].iloc[0]) + 1e-9, 3)


def rebuild_fold_labels():
    """Test labels of each fold, exactly as prepare_loo_folds_upsample.py made them."""
    df = pd.read_csv(os.path.join(UP, 'cleaned_data', 'AID_1347037_cleaned.csv'),
                     usecols=['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME'])

    def canon(s):
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m else None

    df['csmi'] = df['PUBCHEM_EXT_DATASOURCE_SMILES'].apply(canon)
    df = df.dropna(subset=['csmi']).reset_index(drop=True)
    df = df[df['PUBCHEM_ACTIVITY_OUTCOME'].isin(['Active', 'Inactive'])]
    # every remaining SMILES parses, so the fingerprint-validity filter in the
    # original script removes nothing
    df = df[df['csmi'].apply(lambda s: Chem.MolFromSmiles(s) is not None)].reset_index(drop=True)
    y = (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return y, [y[test] for _, test in skf.split(np.zeros(len(y)), y)]


def check_against_saved(labels):
    """Recomputed per-fold ROC-AUC must match the saved per-fold metrics."""
    saved = {
        'Consensus': pd.read_csv(os.path.join(GT_CORR, 'consensus_metrics.csv')),
        'GIN': pd.read_csv(os.path.join(UP, 'results_loo', 'gin_loo_per_fold.csv')),
    }
    for name, df in saved.items():
        fold_col = 'Fold' if 'Fold' in df.columns else df.columns[0]
        df = df[(df['Model'] == name) & (df['CPT'] == 0.5)]
        if 'Split' in df.columns:
            df = df[df['Split'] == 'Test']
        for f in range(N_FOLDS):
            p = np.load(MODELS[name][0].format(f=f))
            mine = roc_auc_score(labels[f], p)
            ref = float(df[df[fold_col] == f]['ROC-AUC'].iloc[0])
            if abs(mine - ref) > 1e-3:
                raise SystemExit(f'{name} fold {f}: rebuilt labels give AUC {mine:.4f}, '
                                 f'saved {ref:.4f} -- labels do not match, not plotting')
    print('Rebuilt fold labels reproduce the saved per-fold ROC-AUCs.')


def main():
    y_all, labels = rebuild_fold_labels()
    print(f'Compounds: {len(y_all):,}  active={y_all.sum()}  inactive={(y_all == 0).sum():,}')
    check_against_saved(labels)

    grid = np.linspace(0, 1, 501)
    curves = {}
    for name, (pattern, *_ ) in MODELS.items():
        tprs, precs, aucs, aps = [], [], [], []
        for f in range(N_FOLDS):
            y, p = labels[f], np.load(pattern.format(f=f))
            fpr, tpr, _ = roc_curve(y, p)
            t = np.interp(grid, fpr, tpr); t[0] = 0.0
            tprs.append(t)
            pre, rec, _ = precision_recall_curve(y, p)
            # recall is decreasing in sklearn's output; interpolate the
            # step-wise (max-precision-to-the-right) envelope
            precs.append(np.interp(grid, rec[::-1], np.maximum.accumulate(pre)[::-1]))
            aucs.append(roc_auc_score(y, p))
            aps.append(average_precision_score(y, p))
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        curves[name] = dict(tpr=mean_tpr, prec=np.mean(precs, axis=0),
                            auc=np.mean(aucs), auc_sd=np.std(aucs, ddof=1),
                            ap=np.mean(aps), ap_sd=np.std(aps, ddof=1))
        print(f'{name:14s} ROC-AUC {curves[name]["auc"]:.3f} ± {curves[name]["auc_sd"]:.3f}   '
              f'PR-AUC {curves[name]["ap"]:.3f} ± {curves[name]["ap_sd"]:.3f}')

    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': '#52514e',
                         'axes.linewidth': 0.8})

    # ROC (left) and PR (right) side by side in one figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    for name, (_, color, ls, lw) in MODELS.items():
        c = curves[name]
        ax1.plot(grid, c['tpr'], color=color, ls=ls, lw=lw,
                 label=f'{name} ({table_value(name, "ROC-AUC"):.3f})')
        ax2.plot(grid, c['prec'], color=color, ls=ls, lw=lw,
                 label=f'{name} ({table_value(name, "PR-AUC"):.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='FPR', ylabel='TPR',
            title='ROC Curves — CHO-K1 (10-fold CV, up-sampling)')
    ax2.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='Recall', ylabel='Precision',
            title='PR Curves — CHO-K1 (10-fold CV, up-sampling)')
    ax1.legend(loc='lower right', fontsize=10, title='Mean ROC-AUC', title_fontsize=10)
    ax2.legend(loc='upper right', fontsize=10, title='Mean PR-AUC', title_fontsize=10)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'chok1_cv_upsampling_roc_pr.{ext}'), dpi=300)
    plt.close(fig)
    print(f'Saved figures to {OUT}')


if __name__ == '__main__':
    main()
