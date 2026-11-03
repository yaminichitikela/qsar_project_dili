"""
Builds 5 repeated Scaffold splits, shared by every scaffold model job.
Run this first (CPU only) -- every train_*_scaffold.py script loads its
input from here, so every model uses the exact same 5 scaffold splits.
 "Molecules are partitioned into bins based
on their Murcko scaffold calculated by RDKit. Any bins larger than half of
the desired test set size are placed into the training set, in order to
guarantee the scaffold diversity of the validation and test sets. All
remaining bins are placed randomly into the training, validation, and test
sets until each set has reached its desired size. As this latter process
involves randomly placing scaffolds into bins, we are able to generate
several different scaffold splits for evaluation."

 "We apply a 5 fold cross-validation here with 8:1:1
train/validation/test split too, and calculate the mean and standard
deviation as the final results." -- so 5 repeats, 8:1:1 target ratio, a
fresh shuffle of the (non-oversized) scaffold bins each repeat.

Downsampling (to 1:1) is applied to the training set only, same as the
random-split and leave-one-out scripts. Splits are NOT stratified (per
Yang et al.: "None of our splits on classification datasets are
stratified; we do not enforce class balance.") -- only the bin-size rule
and the fill order (test -> val -> train) determine class balance here.

Usage: python prepare_scaffold_upsample.py
Outputs (written to ./data_scaffold/repeat_<i>/):
    X_train.npy, y_train.npy   -- up-sampled 1:1 training fingerprints
    X_val.npy,   y_val.npy     -- scaffold-split validation fingerprints
    X_test.npy,  y_test.npy    -- scaffold-split test fingerprints
    graphs_train.pt, graphs_val.pt, graphs_test.pt  -- same split, as PyG graphs

UP-SAMPLED VERSION, Caspase-3/7 HepG2, AID 1347034 ONLY (no AID 654 merge).
The split logic is identical to ../modeling/wahab_jobs/prepare_scaffold_splits.py (same SEED,
same assignment rules, same features); only the input is the standalone
cleaned_data/AID_1347034_cleaned.csv. There is no published downsampled run on
this dataset, so these val/test sets are new (they differ from the merged
AID 654 + 1347034 runs). Balancing: actives in the TRAINING set are duplicated
with replacement until they match the inactive count, and every inactive is
kept. Val/test keep the natural ratio and are never resampled.
"""
import random
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdchem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

SEED = 42
N_REPEATS = 5
FRAC_VAL = 0.10
FRAC_TEST = 0.10

CLEANED_CSV = os.path.join(os.path.dirname(__file__), 'cleaned_data', 'AID_1347034_cleaned.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data_scaffold')
os.makedirs(OUT_DIR, exist_ok=True)


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def murcko_scaffold(smi):
    csmi = canonicalize(smi)
    if csmi is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(csmi)
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


def yang_scaffold_split(scaffolds, n_total, seed):
    """Yang et al. scaffold split: bins larger than half the test-set target
    go straight to train; remaining bins are shuffled and fill test -> val ->
    train in order. Returns (train_rows, val_rows, test_rows, n_big_bins)."""
    target_test = int(FRAC_TEST * n_total)
    target_val = int(FRAC_VAL * n_total)
    big_bin_cutoff = target_test / 2

    bin_to_rows = {}
    for row, scaf in enumerate(scaffolds):
        bin_to_rows.setdefault(scaf, []).append(row)

    big_bins = [rows for rows in bin_to_rows.values() if len(rows) > big_bin_cutoff]
    small_bins = [list(rows) for rows in bin_to_rows.values() if len(rows) <= big_bin_cutoff]

    rng = random.Random(seed)
    rng.shuffle(small_bins)

    train_rows = [i for rows in big_bins for i in rows]
    val_rows, test_rows = [], []
    for rows in small_bins:
        if len(test_rows) < target_test:
            test_rows += rows
        elif len(val_rows) < target_val:
            val_rows += rows
        else:
            train_rows += rows

    return train_rows, val_rows, test_rows, len(big_bins)


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
    n_total = len(df)

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

    # ── scaffolds, computed once (reused every repeat -- only the bin fill
    #    order is reshuffled per repeat) ────────────────────
    scaffolds = df['csmi'].apply(murcko_scaffold).values
    print(f'Distinct Murcko scaffolds: {len(set(scaffolds)):,}')

    for r in range(N_REPEATS):
        seed = SEED + r
        train_rows, val_rows, test_rows, n_big = yang_scaffold_split(scaffolds, n_total, seed)
        train_idx_raw = np.array(train_rows)
        val_idx = np.array(val_rows)
        test_idx = np.array(test_rows)

        # up-sample train only, to 1:1
        y_tr_raw = y_all[train_idx_raw]
        act_idx = np.where(y_tr_raw == 1)[0]
        ina_idx = np.where(y_tr_raw == 0)[0]
        rng = np.random.RandomState(seed)
        sampled_act = rng.choice(act_idx, size=len(ina_idx), replace=True)
        keep = np.concatenate([sampled_act, ina_idx])
        rng.shuffle(keep)
        train_idx = train_idx_raw[keep]

        rep_dir = os.path.join(OUT_DIR, f'repeat_{r}')
        os.makedirs(rep_dir, exist_ok=True)

        X_train, y_train = X_fp[train_idx], y_all[train_idx]
        X_val, y_val = X_fp[val_idx], y_all[val_idx]
        X_test, y_test = X_fp[test_idx], y_all[test_idx]

        np.save(os.path.join(rep_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(rep_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(rep_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(rep_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(rep_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(rep_dir, 'y_test.npy'), y_test)

        torch.save(graphs_for(train_idx), os.path.join(rep_dir, 'graphs_train.pt'))
        torch.save(graphs_for(val_idx), os.path.join(rep_dir, 'graphs_val.pt'))
        torch.save(graphs_for(test_idx), os.path.join(rep_dir, 'graphs_test.pt'))

        print(f'Repeat {r} (big bins forced to train: {n_big}): '
              f'train={len(train_idx)} (act={(y_train==1).sum()}, ina={(y_train==0).sum()}, up-sampled 1:1 from {len(act_idx)} training actives)  '
              f'val={len(val_idx)} (act={(y_val==1).sum()}, ina={(y_val==0).sum()})  '
              f'test={len(test_idx)} (act={(y_test==1).sum()}, ina={(y_test==0).sum()})')

    print(f'\nSaved all {N_REPEATS} scaffold-split repeats to {OUT_DIR}/repeat_0 .. repeat_{N_REPEATS-1}')


if __name__ == '__main__':
    main()
