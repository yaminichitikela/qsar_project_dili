"""Builds a 5-fold cross-validation split over the FULL InChIKey-cleaned
clinical pool, for ALL 10 models (RF/SVM/XGB, GCN/GAT/GIN/SAGE, Graphormer/
GraphGPS, MV-GNN) -- extends the MV-GNN-only version to build every
representation prepare_proxy_data_clean.py (DILI_proxy_clean) builds for a
single split, now per-fold instead:
    X_<split>.npy, y_<split>.npy   -- ECFP4 1024-bit (classical: RF/SVM/XGB)
    atom_<split>.pt, bond_<split>.pt -- MV-GNN graph pairs
    gt_<split>.pt                  -- single graph + Laplacian PE
                                       (GCN/GAT/GIN/SAGE and Graphormer/GraphGPS)

Same two real bugs fixed vs. the original (MV-GNN-only) prepare_dili_cv_data.py:
  1. csmi-string dedup -> InChIKey dedup with majority-vote label resolution.
  2. Clinical pool only excluded Gold-1111 -> now excludes Gold-1111 AND
     Held-223 AND NCV-888 (by InChIKey), so no fold's training data can
     contain compounds from the shared held-out test set or NCV-888.

5-fold StratifiedKFold over the FULL cleaned clinical pool (no discard).

Usage: python prepare_dili_cv_data.py
Outputs (written to ./data/):
    fold_<i>/{X,atom,bond,gt}_train.*, {X,atom,bond,gt}_val.*, y_train.npy, y_val.npy  for i in 0..4
    {X,atom,bond,gt}_ncv.*,  y_ncv.npy   -- NCV-888, shared by all folds (Stage-2 only)
    {X,atom,bond,gt}_test.*, y_test.npy  -- held-out-223, shared by all folds
"""
import os
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, inchi
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj

RDLogger.DisableLog('rdApp.*')

SEED = 42
N_FOLDS = 5
N_BITS = 1024
PE_DIM = 8
np.random.seed(SEED)
torch.manual_seed(SEED)

LIVER_DIR = os.path.join(os.path.dirname(__file__), 'dili_data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUT_DIR, exist_ok=True)


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def smiles_to_inchikey(smi):
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol) or None
    except Exception:
        return None


ATOM_SYMBOLS = [6, 7, 8, 16, 9, 15, 17, 35, 53]
BOND_DIM = 6


def one_hot(val, choices):
    v = [0] * len(choices)
    try:
        v[choices.index(val)] = 1
    except ValueError:
        pass
    return v


def atom_features(atom):
    return (one_hot(atom.GetAtomicNum(), ATOM_SYMBOLS) + [1]
            + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
            + [atom.GetFormalCharge()]
            + one_hot(int(atom.GetHybridization()), [2, 3, 4, 5])
            + [int(atom.GetIsAromatic()), int(atom.IsInRing())])


def bond_features(bond):
    bt = bond.GetBondTypeAsDouble()
    return [float(bt == 1.0), float(bt == 1.5), float(bt == 2.0), float(bt == 3.0),
            float(bond.GetIsConjugated()), float(bond.IsInRing())]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles('C').GetAtomWithIdx(0)))


def mol_to_atom_graph(smi, label):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    if mol.GetNumBonds() == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_DIM), dtype=torch.float)
    else:
        edges, attrs = [], []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bf = bond_features(b)
            edges += [[i, j], [j, i]]
            attrs += [bf, bf]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.float))


def mol_to_bond_graph(smi, label):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    bonds = list(mol.GetBonds())
    if len(bonds) == 0:
        x = torch.zeros((1, BOND_DIM), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, ATOM_DIM), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor([label], dtype=torch.float))
    x = torch.tensor([bond_features(b) for b in bonds], dtype=torch.float)
    atom_to_bonds = {}
    for bi, b in enumerate(bonds):
        for a in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()):
            atom_to_bonds.setdefault(a, []).append(bi)
    edges, attrs = [], []
    for atom_idx, bond_list in atom_to_bonds.items():
        if len(bond_list) < 2:
            continue
        af = atom_features(mol.GetAtomWithIdx(atom_idx))
        for i in range(len(bond_list)):
            for j in range(len(bond_list)):
                if i == j:
                    continue
                edges.append([bond_list[i], bond_list[j]])
                attrs.append(af)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attrs, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, ATOM_DIM), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.float))


def compute_laplacian_pe(data, k=PE_DIM):
    n = data.num_nodes
    if n <= 2:
        return torch.zeros(n, k)
    ei, ew = get_laplacian(data.edge_index, normalization='sym', num_nodes=n)
    L = to_dense_adj(ei, edge_attr=ew, max_num_nodes=n).squeeze(0).numpy()
    try:
        import numpy.linalg as nla
        _, vecs = nla.eigh(L)
        num_take = min(k, n - 1)
        pe = torch.from_numpy(vecs[:, 1:1 + num_take]).float()
        if pe.shape[1] < k:
            pe = torch.nn.functional.pad(pe, (0, k - pe.shape[1]))
    except Exception:
        pe = torch.zeros(n, k)
    return pe


def smiles_to_ecfp4(smi, radius=2, n_bits=N_BITS):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_and_tag(path, usecols=('smiles_r', 'TOXICITY')):
    df = pd.read_csv(path, usecols=list(usecols))
    df['csmi'] = df['smiles_r'].apply(canonicalize)
    df['inchikey'] = df['smiles_r'].apply(smiles_to_inchikey)
    df = df.dropna(subset=['csmi', 'inchikey']).reset_index(drop=True)
    return df.rename(columns={'TOXICITY': 'label'})


def dedup_by_inchikey_majority(df, label_col='label'):
    """Collapses each InChIKey group to one row. Agreement -> keep; conflict
    -> majority vote; exact tie -> drop (ambiguous, no safe default)."""
    kept_rows = []
    n_conflict_resolved, n_conflict_dropped = 0, 0
    for ikey, g in df.groupby('inchikey'):
        if len(g) == 1 or g[label_col].nunique() == 1:
            kept_rows.append(g.iloc[0])
            continue
        counts = g[label_col].value_counts()
        if len(counts) == 2 and counts.iloc[0] == counts.iloc[1]:
            n_conflict_dropped += 1
            continue
        majority_label = counts.idxmax()
        row = g.iloc[0].copy()
        row[label_col] = majority_label
        kept_rows.append(row)
        n_conflict_resolved += 1
    out = pd.DataFrame(kept_rows).reset_index(drop=True)
    print(f'  InChIKey dedup: {len(df):,} rows -> {len(out):,} unique compounds '
          f'({n_conflict_resolved} label conflicts resolved by majority vote, '
          f'{n_conflict_dropped} dropped on exact tie)')
    return out


def build_all_features(df, name):
    labels = df['label'].values.astype(int)
    smis = df['csmi'].values
    X_list, atom_list, bond_list, gt_list, y_list = [], [], [], [], []
    for smi, lbl in zip(smis, labels):
        fp = smiles_to_ecfp4(smi)
        ag = mol_to_atom_graph(smi, int(lbl))
        bg = mol_to_bond_graph(smi, int(lbl))
        if fp is None or ag is None or bg is None:
            continue
        gt_graph = Data(x=ag.x, edge_index=ag.edge_index, edge_attr=ag.edge_attr,
                         y=ag.y, pe=compute_laplacian_pe(ag))
        X_list.append(fp)
        atom_list.append(ag)
        bond_list.append(bg)
        gt_list.append(gt_graph)
        y_list.append(int(lbl))
    y = np.array(y_list)
    print(f'{name}: {len(y):,}  DILI+={int(y.sum()):,}  DILI-={int((y == 0).sum()):,}')
    return np.vstack(X_list), y, atom_list, bond_list, gt_list


def save_split(out_dir, name, X, y, atom_g, bond_g, gt_g):
    np.save(os.path.join(out_dir, f'X_{name}.npy'), X)
    np.save(os.path.join(out_dir, f'y_{name}.npy'), y)
    torch.save(atom_g, os.path.join(out_dir, f'atom_{name}.pt'))
    torch.save(bond_g, os.path.join(out_dir, f'bond_{name}.pt'))
    torch.save(gt_g, os.path.join(out_dir, f'gt_{name}.pt'))


def main():
    print('=== Loading + computing InChIKey for all 4 DILI files ===')
    gold_df = load_and_tag(f'{LIVER_DIR}/DILI_Goldstandard_1111.csv')
    held_df = load_and_tag(f'{LIVER_DIR}/test_data_heldouttest_DILIst_223.csv')
    all_df = load_and_tag(f'{LIVER_DIR}/Alloriginaldata_19911.csv')
    ncv_df = load_and_tag(f'{LIVER_DIR}/test_data_ncv_DILIst_888.csv')

    print('\n=== Deduping each file by InChIKey (majority vote on label conflicts) ===')
    print('Gold-1111:');    gold_df = dedup_by_inchikey_majority(gold_df)
    print('Held-223:');     held_df = dedup_by_inchikey_majority(held_df)
    print('Alloriginal:');  all_df = dedup_by_inchikey_majority(all_df)
    print('NCV-888:');      ncv_df = dedup_by_inchikey_majority(ncv_df)

    exclude_ikeys = (set(gold_df['inchikey']) | set(held_df['inchikey'])
                      | set(ncv_df['inchikey']))
    clin_df = all_df[~all_df['inchikey'].isin(exclude_ikeys)].reset_index(drop=True)
    print(f'\nClinical pool (Alloriginal - Gold-1111 - Held-223 - NCV-888, by InChIKey): '
          f'{len(clin_df):,}  DILI+={clin_df.label.sum():,}  DILI-={(clin_df.label == 0).sum():,}')

    n_leak_held = clin_df['inchikey'].isin(set(held_df['inchikey'])).sum()
    n_leak_ncv = clin_df['inchikey'].isin(set(ncv_df['inchikey'])).sum()
    assert n_leak_held == 0, f'Held-223 still leaking into clinical pool: {n_leak_held}'
    assert n_leak_ncv == 0, f'NCV-888 still leaking into clinical pool: {n_leak_ncv}'
    print('Leakage check (InChIKey): clinical pool has 0 overlap with Held-223, 0 overlap with NCV-888.')

    print('\n=== Building shared held-out-223 features (ECFP4 + atom/bond + gt) ===')
    X_te, y_te, atom_te, bond_te, gt_te = build_all_features(held_df, 'held_test (223, shared)')
    save_split(OUT_DIR, 'test', X_te, y_te, atom_te, bond_te, gt_te)

    print('\n=== Building shared NCV-888 features (Stage-2 only) ===')
    X_ncv, y_ncv, atom_ncv, bond_ncv, gt_ncv = build_all_features(ncv_df, 'NCV-888 (shared)')
    save_split(OUT_DIR, 'ncv', X_ncv, y_ncv, atom_ncv, bond_ncv, gt_ncv)

    print('\n=== Building 5-fold CV splits over the full clinical pool ===')
    y_clin = clin_df['label'].values.astype(int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_clin)), y_clin)):
        fold_dir = os.path.join(OUT_DIR, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        train_df = clin_df.iloc[tr_idx].reset_index(drop=True)
        val_df = clin_df.iloc[va_idx].reset_index(drop=True)

        X_tr, y_tr, atom_tr, bond_tr, gt_tr = build_all_features(train_df, f'fold_{fold} train')
        X_va, y_va, atom_va, bond_va, gt_va = build_all_features(val_df, f'fold_{fold} val')

        save_split(fold_dir, 'train', X_tr, y_tr, atom_tr, bond_tr, gt_tr)
        save_split(fold_dir, 'val', X_va, y_va, atom_va, bond_va, gt_va)

    print(f'\nAtom feature dim: {ATOM_DIM}  Bond feature dim: {BOND_DIM}  ECFP4 bits: {N_BITS}')
    print(f'Saved {N_FOLDS} folds + shared NCV-888 + shared held-out-223 to {OUT_DIR}/')


if __name__ == '__main__':
    main()
