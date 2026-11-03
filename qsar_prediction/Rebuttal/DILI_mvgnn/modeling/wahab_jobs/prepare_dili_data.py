"""Builds the DILI clin_train / clin_val / held-out-223 split, exactly matching
proxy_to_dili_consistent copy 2.ipynb cells 1, 4, 6, 9 (same canonicalization,
same clin_df = Alloriginal - Gold1111 definition, same 80/10/[10 discarded]
split, same held-out-223 test set, same 23-dim atom / 6-dim bond graph
features used there by Graphormer/GraphGPS) -- so MV-GNN^cross is evaluated
on identical data/splits/features as the notebook's other DILI models, for a
fair comparison in Table 6.

Builds BOTH an atom-oriented graph (standard: x=atom feats, edge_index/edge_attr
from bonds) and a bond-oriented graph (bonds-as-nodes; two bonds are adjacent if
they share an atom) per molecule, since MV-GNN^cross needs both views.

Usage: python prepare_dili_data.py
Outputs (written to ./data/):
    atom_train.pt, atom_val.pt, atom_test.pt   -- atom-oriented PyG graphs
    bond_train.pt, bond_val.pt, bond_test.pt   -- bond-oriented PyG graphs
    y_train.npy, y_val.npy, y_test.npy
"""
import os
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import torch
from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import pandas as pd

RDLogger.DisableLog('rdApp.*')

SEED = 42
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


# ── atom / bond features, identical to proxy_to_dili_consistent copy 2.ipynb cell 6 ──
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
    """Standard atom-oriented graph: nodes=atoms, edges=bonds."""
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
    """Bond-oriented graph (paper's 'line graph' view): nodes=bonds, node
    features=bond_features; two bond-nodes are connected if they share an
    atom, edge feature=that shared atom's atom_features."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    bonds = list(mol.GetBonds())
    if len(bonds) == 0:
        # no bonds: single dummy bond-node with zero features
        x = torch.zeros((1, BOND_DIM), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, ATOM_DIM), dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor([label], dtype=torch.float))

    x = torch.tensor([bond_features(b) for b in bonds], dtype=torch.float)

    # map: atom idx -> list of bond indices incident to it
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


def load_and_clean(path, usecols=('smiles_r', 'TOXICITY')):
    df = pd.read_csv(path, usecols=list(usecols))
    df['csmi'] = df['smiles_r'].apply(canonicalize)
    df = (df.dropna(subset=['csmi'])
            .drop_duplicates('csmi')
            .reset_index(drop=True)
            [['csmi', 'TOXICITY']].rename(columns={'TOXICITY': 'label'}))
    return df


def build_graphs(df, name):
    labels = df['label'].values.astype(int)
    atom_graphs, bond_graphs, keep = [], [], []
    for smi, lbl in zip(df['csmi'].values, labels):
        ag = mol_to_atom_graph(smi, int(lbl))
        bg = mol_to_bond_graph(smi, int(lbl))
        if ag is not None and bg is not None:
            atom_graphs.append(ag)
            bond_graphs.append(bg)
            keep.append(True)
        else:
            keep.append(False)
    y = labels[np.array(keep)]
    print(f'{name}: {len(y):,}  DILI+={int(y.sum()):,}  DILI-={int((y==0).sum()):,}')
    return atom_graphs, bond_graphs, y


def main():
    # ── Gold-1111 exclusion set (cell 4) ──────────────────────────────────
    gold_df = load_and_clean(f'{LIVER_DIR}/DILI_Goldstandard_1111.csv')
    gold_smiles = set(gold_df['csmi'])
    print(f'Gold Standard 1111: {len(gold_smiles):,} compounds')

    # ── Held-out-223 (final test) ──────────────────────────────────────────
    held_df = load_and_clean(f'{LIVER_DIR}/test_data_heldouttest_DILIst_223.csv')
    print(f'Held-223  : {len(held_df):,}  DILI+={held_df.label.sum():,}  '
          f'DILI-={(held_df.label==0).sum():,}')

    # ── Clinical training set: Alloriginal - Gold-1111 (cell 4) ────────────
    all_df = load_and_clean(f'{LIVER_DIR}/Alloriginaldata_19911.csv')
    clin_df = all_df[~all_df['csmi'].isin(gold_smiles)].reset_index(drop=True)
    print(f'Clinical (Alloriginal - Gold-1111) : {len(clin_df):,}  '
          f'DILI+={clin_df.label.sum():,}  DILI-={(clin_df.label==0).sum():,}')

    # ── Clinical 80/10/[10 discarded] split (cell 9, replicated exactly) ───
    y_clin = clin_df['label'].values.astype(int)
    idx_c = np.arange(len(y_clin))
    idx_c_pool, _ = train_test_split(idx_c, test_size=0.10, stratify=y_clin, random_state=SEED)
    idx_c_tr, idx_c_va = train_test_split(
        idx_c_pool, test_size=0.10 / 0.90, stratify=y_clin[idx_c_pool], random_state=SEED)

    clin_train_df = clin_df.iloc[idx_c_tr].reset_index(drop=True)
    clin_val_df = clin_df.iloc[idx_c_va].reset_index(drop=True)
    print(f'Clinical  train / val : {len(idx_c_tr):,} / {len(idx_c_va):,}  '
          f'(remaining {len(idx_c) - len(idx_c_pool):,} discarded, matches notebook)')

    # ── build atom-oriented + bond-oriented graphs for each split ──────────
    atom_tr, bond_tr, y_tr = build_graphs(clin_train_df, 'clin_train')
    atom_va, bond_va, y_va = build_graphs(clin_val_df, 'clin_val')
    atom_te, bond_te, y_te = build_graphs(held_df, 'held_test (223)')

    torch.save(atom_tr, os.path.join(OUT_DIR, 'atom_train.pt'))
    torch.save(atom_va, os.path.join(OUT_DIR, 'atom_val.pt'))
    torch.save(atom_te, os.path.join(OUT_DIR, 'atom_test.pt'))
    torch.save(bond_tr, os.path.join(OUT_DIR, 'bond_train.pt'))
    torch.save(bond_va, os.path.join(OUT_DIR, 'bond_val.pt'))
    torch.save(bond_te, os.path.join(OUT_DIR, 'bond_test.pt'))
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_tr)
    np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_va)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_te)

    print(f'\nAtom feature dim: {ATOM_DIM}  Bond feature dim: {BOND_DIM}')
    print(f'Saved all split files to {OUT_DIR}/')


if __name__ == '__main__':
    main()
