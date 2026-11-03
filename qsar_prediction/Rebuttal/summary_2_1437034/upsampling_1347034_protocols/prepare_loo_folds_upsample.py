"""
Builds the Leave-one-out 10-fold split, shared by every LOO model job.
Run this first (CPU only) -- every train_*_loo.py script loads its input
from here, so every model uses the exact same 10 folds.

Per Ma et al.: "The input dataset is split into 10 folds equally, each fold
has been used as the testing dataset in sequence. Within the remaining 9
folds, one fold is used as the validation dataset, and the rest are used
for training."

Fold i: test = fold i, validation = fold (i+1) mod 10, train = remaining 8
folds, up-sampled to 1:1 (train only -- val/test stay at natural ratio).

Usage: python prepare_loo_folds_upsample.py
Outputs (written to ./data_loo/fold_<i>/):
    X_train.npy, y_train.npy   -- up-sampled 1:1 training fingerprints
    X_val.npy,   y_val.npy     -- natural-imbalance validation fingerprints
    X_test.npy,  y_test.npy    -- natural-imbalance test fingerprints
    graphs_train.pt, graphs_val.pt, graphs_test.pt  -- same split, as PyG graphs

UP-SAMPLED VERSION, Caspase-3/7 HepG2, AID 1347034 ONLY (no AID 654 merge).
The split logic is identical to ../modeling/wahab_jobs/prepare_loo_folds.py (same SEED,
same assignment rules, same features); only the input is the standalone
cleaned_data/AID_1347034_cleaned.csv. There is no published downsampled run on
this dataset, so these val/test sets are new (they differ from the merged
AID 654 + 1347034 runs). Balancing: actives in the TRAINING set are duplicated
with replacement until they match the inactive count, and every inactive is
kept. Val/test keep the natural ratio and are never resampled.
"""
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdchem
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data

SEED = 42
N_FOLDS = 10
np.random.seed(SEED)
torch.manual_seed(SEED)

CLEANED_CSV = os.path.join(os.path.dirname(__file__), 'cleaned_data', 'AID_1347034_cleaned.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data_loo')
os.makedirs(OUT_DIR, exist_ok=True)


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def smiles_to_ecfp4(smi, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


ATOM_SYMBOLS = [6, 7, 8, 16, 9, 15, 17, 35, 53]


def one_hot(val, choices):
    vec = [0] * len(choices)
    try:
        vec[choices.index(val)] = 1
    except ValueError:
        pass
    return vec


HYBRID_MAP = {
    rdchem.HybridizationType.SP: 0,
    rdchem.HybridizationType.SP2: 1,
    rdchem.HybridizationType.SP3: 2,
}


def atom_feat(atom):
    at_oh = one_hot(atom.GetAtomicNum(), ATOM_SYMBOLS) + [0]
    hyb = [0, 0, 0, 0]
    hyb[HYBRID_MAP.get(atom.GetHybridization(), 3)] = 1
    deg = min(atom.GetDegree(), 5)
    deg_oh = [0] * 6
    deg_oh[deg] = 1
    fc = atom.GetFormalCharge()
    fc_enc = [1 if fc < 0 else 0, 1 if fc == 0 else 0, 1 if fc > 0 else 0]
    extras = [
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        min(atom.GetTotalNumHs(), 4) / 4.0,
        atom.GetMass() / 100.0,
    ]
    return at_oh + deg_oh + fc_enc + hyb + extras  # 27-dim


BOND_TYPES = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
              rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]


def bond_feat(bond):
    bt = one_hot(bond.GetBondType(), BOND_TYPES)
    return bt + [int(bond.GetIsConjugated()), int(bond.IsInRing())]  # 6-dim


def mol_to_graph(smi, label):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    x = torch.tensor([atom_feat(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_feat(bond)
        ei += [[i, j], [j, i]]
        ea += [bf, bf]
    if ei:
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(ea, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def main():
    # ── load cleaned data, label ────────────────────────────────────────────
    df_raw = pd.read_csv(CLEANED_CSV, usecols=[
        'PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES',
        'PUBCHEM_ACTIVITY_SCORE', 'PUBCHEM_ACTIVITY_OUTCOME'])
    df_raw['csmi'] = df_raw['PUBCHEM_EXT_DATASOURCE_SMILES'].apply(canonicalize)
    df = df_raw.dropna(subset=['csmi']).reset_index(drop=True)
    df = df[df['PUBCHEM_ACTIVITY_OUTCOME'].isin(['Active', 'Inactive'])].copy()
    df['label'] = (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)
    print(f'Compounds after cleanup: {len(df):,}  '
          f'(Active={df.label.sum():,}, Inactive={(df.label==0).sum():,})')

    # ── features ─────────────────────────────────────────────────────────────
    fps = df['csmi'].apply(smiles_to_ecfp4)
    valid = fps.notna()
    df = df[valid].reset_index(drop=True)
    X_fp = np.vstack(fps[valid].values)
    y_all = df['label'].values

    graph_list, valid_g = [], []
    for i, (smi, lbl) in enumerate(zip(df['csmi'].values, y_all)):
        g = mol_to_graph(smi, lbl)
        if g is not None:
            graph_list.append(g)
            valid_g.append(i)
    valid_g = np.array(valid_g)
    orig_to_graph_pos = {orig: pos for pos, orig in enumerate(valid_g)}
    print(f'Graphs built: {len(graph_list):,}')

    def graphs_for(idx_arr):
        return [graph_list[orig_to_graph_pos[i]] for i in idx_arr if i in orig_to_graph_pos]

    # ── 10 equal folds (fold i = test, fold i+1 = validation, rest = train) ──
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = [test_idx for _, test_idx in skf.split(X_fp, y_all)]

    for i in range(N_FOLDS):
        test_idx = folds[i]
        val_idx = folds[(i + 1) % N_FOLDS]
        train_idx_raw = np.concatenate([folds[j] for j in range(N_FOLDS) if j not in (i, (i + 1) % N_FOLDS)])

        y_tr_raw = y_all[train_idx_raw]
        act_idx = np.where(y_tr_raw == 1)[0]
        ina_idx = np.where(y_tr_raw == 0)[0]
        rng = np.random.RandomState(SEED + i)
        sampled_act = rng.choice(act_idx, size=len(ina_idx), replace=True)
        keep = np.concatenate([sampled_act, ina_idx])
        rng.shuffle(keep)
        train_idx = train_idx_raw[keep]

        fold_dir = os.path.join(OUT_DIR, f'fold_{i}')
        os.makedirs(fold_dir, exist_ok=True)

        X_train, y_train = X_fp[train_idx], y_all[train_idx]
        X_val, y_val = X_fp[val_idx], y_all[val_idx]
        X_test, y_test = X_fp[test_idx], y_all[test_idx]

        np.save(os.path.join(fold_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(fold_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(fold_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(fold_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(fold_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(fold_dir, 'y_test.npy'), y_test)

        torch.save(graphs_for(train_idx), os.path.join(fold_dir, 'graphs_train.pt'))
        torch.save(graphs_for(val_idx), os.path.join(fold_dir, 'graphs_val.pt'))
        torch.save(graphs_for(test_idx), os.path.join(fold_dir, 'graphs_test.pt'))

        print(f'Fold {i}: train={len(train_idx)} (act={(y_train==1).sum()}, ina={(y_train==0).sum()}, up-sampled 1:1 from {len(act_idx)} training actives)  '
              f'val={len(val_idx)} (act={(y_val==1).sum()}, ina={(y_val==0).sum()})  '
              f'test={len(test_idx)} (act={(y_test==1).sum()}, ina={(y_test==0).sum()})')

    print(f'\nSaved all 10 folds to {OUT_DIR}/fold_0 .. fold_9')


if __name__ == '__main__':
    main()
