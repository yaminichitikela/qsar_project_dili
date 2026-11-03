"""Applicability domain for the caspase-3/7 summary assays, up-sampled 10-fold CV.

Reviewer 1, Comment 14. Replaces SI Figure S1, which was built from the
down-sampled models and the merged AID 654+1347034 HepG2 set.

Datasets (single summary assays, the ones the main-text up-sampled tables use):
    HepG2  AID 1347034  summary_2_1437034/upsampling_1347034_protocols
    CHO-K1 AID 1347037  summary_1347037/upsampling

Folds are rebuilt exactly as prepare_loo_folds_upsample.py does (same cleaning,
SEED, StratifiedKFold; test = fold i, val = fold i+1, train = the other 8).

AD reference set = the UNIQUE training compounds of each fold. Up-sampling only
duplicates actives, which adds no chemical space, and duplicates would put
spurious zero distances into the training distribution. Training-compound
distances exclude the compound itself (6 neighbours, first dropped); test
compounds use their 5 nearest training compounds. Threshold = training mean +
k * SD (k = 1.5 in the manuscript). Predictions = full 9-model consensus.

Usage: python make_ad_caspase_upsampled.py
Outputs: updated_figures/ad_histogram_caspase_upsampled.png
         results/ad_caspase_upsampled_summary.csv  (inside/outside AD metrics per k)
"""
import os
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy import stats

RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
N_FOLDS = 10
K_PRIMARY = 1.5
K_SWEEP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CONSENSUS = ['rf', 'svm', 'xgb', 'gcn', 'gat', 'gin', 'sage', 'graphormer', 'graphgps']

ASSAYS = {
    'HepG2 (AID 1347034)': f'{HERE}/summary_2_1437034/upsampling_1347034_protocols',
    'CHO-K1 (AID 1347037)': f'{HERE}/summary_1347037/upsampling',
}
# CHO-K1's reported consensus uses the corrected graph transformers, so its
# full-consensus probabilities are read from graph_transformer_corrected
# (ROC-AUC 0.837, as in the CHO-K1 CV table). HepG2's upsampling_1347034_protocols
# run already trained the corrected GTs, so its 9-model mean reproduces the SI
# table (0.890) directly.
CORRECTED_CONSENSUS = {
    'CHO-K1 (AID 1347037)': f'{HERE}/graph_transformer_corrected/results/chok1/up_loo/probs/Consensus_{{i}}_test_probs.npy',
}
CSV = {
    'HepG2 (AID 1347034)': 'AID_1347034_cleaned.csv',
    'CHO-K1 (AID 1347037)': 'AID_1347037_cleaned.csv',
}


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def ecfp4(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build(root, csv):
    """Same cleaning and fold assignment as prepare_loo_folds_upsample.py."""
    df = pd.read_csv(f'{root}/cleaned_data/{csv}', usecols=[
        'PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES',
        'PUBCHEM_ACTIVITY_SCORE', 'PUBCHEM_ACTIVITY_OUTCOME'])
    df['csmi'] = df['PUBCHEM_EXT_DATASOURCE_SMILES'].apply(canonicalize)
    df = df.dropna(subset=['csmi']).reset_index(drop=True)
    df = df[df['PUBCHEM_ACTIVITY_OUTCOME'].isin(['Active', 'Inactive'])].copy()
    df['label'] = (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)
    fps = df['csmi'].apply(ecfp4)
    valid = fps.notna()
    X = np.vstack(fps[valid].values).astype(bool)
    y = df.loc[valid, 'label'].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = [te for _, te in skf.split(X, y)]
    return X, y, folds


def metrics(y, p):
    if len(y) == 0 or y.min() == y.max():
        return dict(n=len(y), n_act=int(y.sum()), roc_auc=np.nan, pr_auc=np.nan, f1=np.nan)
    return dict(n=len(y), n_act=int(y.sum()), roc_auc=roc_auc_score(y, p),
                pr_auc=average_precision_score(y, p), f1=f1_score(y, (p >= 0.5).astype(int)))


def main():
    rows, pooled = [], {}
    for label, root in ASSAYS.items():
        X, y, folds = build(root, CSV[label])
        print(f'{label}: {len(y):,} compounds, {y.sum()} active')
        tr_all, te_all = [], []
        for i in range(N_FOLDS):
            te = folds[i]
            tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j not in (i, (i + 1) % N_FOLDS)])
            ref = np.unique(X[tr], axis=0)   # unique training fingerprints
            nn = NearestNeighbors(n_neighbors=6, metric='jaccard', n_jobs=-1).fit(ref)
            d_tr = nn.kneighbors(ref)[0][:, 1:].mean(axis=1)          # self excluded
            d_te = nn.kneighbors(X[te], n_neighbors=5)[0].mean(axis=1)
            if label in CORRECTED_CONSENSUS:
                probs = np.load(CORRECTED_CONSENSUS[label].format(i=i))
            else:
                probs = np.mean([np.load(f'{root}/results_loo/probs/{m}_fold{i}_test_probs.npy')
                                 for m in CONSENSUS], axis=0)
            assert len(probs) == len(te), f'{label} fold {i}: {len(probs)} probs vs {len(te)} test'
            yt = y[te]
            tr_all.append(d_tr); te_all.append(d_te)
            for k in K_SWEEP:
                thr = d_tr.mean() + k * d_tr.std()
                inside = d_te <= thr
                rows.append(dict(Assay=label, Fold=i, k=k, threshold=thr, coverage=100 * inside.mean(),
                                 **{f'in_{a}': b for a, b in metrics(yt[inside], probs[inside]).items()},
                                 **{f'out_{a}': b for a, b in metrics(yt[~inside], probs[~inside]).items()},
                                 **{f'all_{a}': b for a, b in metrics(yt, probs).items()}))
        pooled[label] = (np.concatenate(tr_all), np.concatenate(te_all))

    df = pd.DataFrame(rows)
    os.makedirs(f'{HERE}/results', exist_ok=True)
    df.to_csv(f'{HERE}/results/ad_caspase_upsampled_per_fold.csv', index=False)

    cols = ['threshold', 'coverage', 'in_roc_auc', 'out_roc_auc', 'all_roc_auc',
            'in_pr_auc', 'out_pr_auc', 'all_pr_auc', 'in_f1', 'out_f1', 'in_n_act', 'out_n_act']
    summ = []
    for (a, k), g in df.groupby(['Assay', 'k']):
        r = {'Assay': a, 'k': k}
        for c in cols:
            v = g[c].dropna().values
            r[c] = v.mean()
            r[c + '_ci'] = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(f'{HERE}/results/ad_caspase_upsampled_summary.csv', index=False)

    # side-by-side histograms, same style as the original SI Figure S1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (label, (d_tr, d_te)) in zip(axes, pooled.items()):
        thr = summ[(summ.Assay == label) & (summ.k == K_PRIMARY)]['threshold'].iloc[0]
        cov = summ[(summ.Assay == label) & (summ.k == K_PRIMARY)]['coverage'].iloc[0]
        bins = np.linspace(0, 1, 51)
        # each compound sits in 8 training sets but 1 test set, so raw counts
        # are ~8x apart; plot proportions so the two shapes are comparable
        ax.hist(d_tr, bins=bins, weights=np.full(len(d_tr), 1 / len(d_tr)),
                alpha=0.6, color='tab:blue', label='Train')
        ax.hist(d_te, bins=bins, weights=np.full(len(d_te), 1 / len(d_te)),
                alpha=0.6, color='salmon', label='Test')
        ax.axvline(thr, color='k', ls='--', lw=1.5, label=f'Threshold={thr:.3f}')
        ax.set_title(f'{label}: {cov:.1f}% of test compounds inside AD')
        ax.set_xlabel('Mean 5-NN Jaccard Distance'); ax.set_ylabel('Fraction of compounds')
        ax.legend(loc='upper left')
    fig.tight_layout()
    os.makedirs(f'{HERE}/updated_figures', exist_ok=True)
    out = f'{HERE}/updated_figures/ad_histogram_caspase_upsampled.png'
    fig.savefig(out, dpi=300)
    print('saved', out)

    pd.set_option('display.width', 220)
    show = summ[['Assay', 'k', 'threshold', 'coverage', 'in_roc_auc', 'out_roc_auc', 'all_roc_auc',
                 'in_pr_auc', 'out_pr_auc', 'in_n_act', 'out_n_act']]
    print(show.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
