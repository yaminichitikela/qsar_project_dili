"""Shared feature-generation and base-split code for the caspase-3/7
sampling-strategy comparison (Reviewer 2's downsampling justification +
Reviewer 1 #13's SMOTE ask, answered as one combined experiment).

This is byte-identical (canonicalization, ECFP4, atom/bond features, the
8:1:1 split via the same SEED=42 train_test_split calls) to each dataset's
own prepare_split.py in summary_2_1437034/ and summary_1347037/, so that:
  - X_val / X_test are IDENTICAL across every variant below and identical to
    the already-trained V0 (random 1:1) baseline -- only the training-set
    construction differs between variants.
  - Every variant is directly comparable to the existing results without
    retraining V0.

Variants live in ./data/<dataset>/<variant>/, in the same X_train.npy /
y_train.npy / graphs_train.pt format the existing train_*.py scripts
already read, so those scripts can be reused unmodified by just pointing
DATA_DIR at a variant folder.
"""
import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdchem
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, 'data')

REBUTTAL_BASE = os.path.dirname(HERE)
CASPASE_DATASETS = {
    'hepg2': {
        'label': 'AID_654+1347034 (Caspase-3/7, HepG2)',
        'cleaned_csv': f'{REBUTTAL_BASE}/summary_2_1437034/cleaning/AID_654_1347034_cleaned.csv',
        'wj': f'{REBUTTAL_BASE}/summary_2_1437034/modeling/wahab_jobs',
    },
    'cho_k1': {
        'label': 'AID_1347037 (Caspase-3/7, CHO-K1)',
        'cleaned_csv': f'{REBUTTAL_BASE}/summary_1347037/cleaning/AID_1347037_cleaned.csv',
        'wj': f'{REBUTTAL_BASE}/summary_1347037/modeling/wahab_jobs',
    },
}


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


def smiles_to_ecfp4_bitvect(smi, radius=2, n_bits=2048):
    """Same fingerprint, kept as an RDKit ExplicitBitVect for fast bulk
    Tanimoto similarity (used by the similarity/diversity variants)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


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


class BaseSplit:
    """Everything shared across variants: fingerprints, graphs, and the
    8:1:1 pool/val/test split -- identical to prepare_split.py."""

    def __init__(self, cleaned_csv):
        df_raw = pd.read_csv(cleaned_csv, usecols=[
            'PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES',
            'PUBCHEM_ACTIVITY_SCORE', 'PUBCHEM_ACTIVITY_OUTCOME'])
        df_raw['csmi'] = df_raw['PUBCHEM_EXT_DATASOURCE_SMILES'].apply(canonicalize)
        df = df_raw.dropna(subset=['csmi']).reset_index(drop=True)
        df = df[df['PUBCHEM_ACTIVITY_OUTCOME'].isin(['Active', 'Inactive'])].copy()
        df['label'] = (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)

        fps = df['csmi'].apply(smiles_to_ecfp4)
        valid = fps.notna()
        df = df[valid].reset_index(drop=True)
        X_fp = np.vstack(fps[valid].values)
        y_all = df['label'].values
        csmi_all = df['csmi'].values

        graph_list, valid_g = [], []
        for i, (smi, lbl) in enumerate(zip(csmi_all, y_all)):
            g = mol_to_graph(smi, lbl)
            if g is not None:
                graph_list.append(g)
                valid_g.append(i)
        valid_g = np.array(valid_g)
        orig_to_graph_pos = {orig: pos for pos, orig in enumerate(valid_g)}

        idx_all = np.arange(len(y_all))
        idx_pool, idx_test = train_test_split(idx_all, test_size=0.10, stratify=y_all, random_state=SEED)
        pool_sub = np.arange(len(idx_pool))
        y_pool = y_all[idx_pool]
        tr_sub, val_sub = train_test_split(pool_sub, test_size=1 / 9, stratify=y_pool, random_state=SEED)
        idx_tr_raw = idx_pool[tr_sub]
        idx_val = idx_pool[val_sub]

        self.X_fp, self.y_all, self.csmi_all = X_fp, y_all, csmi_all
        self.graph_list, self.orig_to_graph_pos = graph_list, orig_to_graph_pos
        self.idx_tr_raw = idx_tr_raw   # raw (pre-downsample) train pool indices
        self.idx_val = idx_val
        self.idx_test = idx_test

    def graphs_for(self, idx_arr):
        return [self.graph_list[self.orig_to_graph_pos[i]] for i in idx_arr if i in self.orig_to_graph_pos]


def save_variant(base: BaseSplit, train_idx, out_dir, extra_X_train=None, extra_y_train=None, skip_graphs=False):
    """Writes one variant's data directory. `train_idx` indexes into
    base.X_fp/base.y_all for the real (non-synthetic) compounds kept in
    this variant's training set; `extra_X_train`/`extra_y_train` are
    appended synthetic fingerprint rows (SMOTE only -- these have no
    matching molecule, so `skip_graphs=True` must be set for that variant:
    graph-based models are not trained on it, see v4_smote_oversample.py).
    """
    os.makedirs(out_dir, exist_ok=True)

    X_train = base.X_fp[train_idx]
    y_train = base.y_all[train_idx]
    if extra_X_train is not None:
        X_train = np.vstack([X_train, extra_X_train])
        y_train = np.concatenate([y_train, extra_y_train])

    X_val, y_val = base.X_fp[base.idx_val], base.y_all[base.idx_val]
    X_test, y_test = base.X_fp[base.idx_test], base.y_all[base.idx_test]

    np.save(os.path.join(out_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(out_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(out_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)

    if not skip_graphs:
        torch.save(base.graphs_for(train_idx), os.path.join(out_dir, 'graphs_train.pt'))
        torch.save(base.graphs_for(base.idx_val), os.path.join(out_dir, 'graphs_val.pt'))
        torch.save(base.graphs_for(base.idx_test), os.path.join(out_dir, 'graphs_test.pt'))

    n_act, n_ina = int((y_train == 1).sum()), int((y_train == 0).sum())
    print(f'  -> {out_dir}')
    print(f'     train={len(y_train):,} (act={n_act:,}, ina={n_ina:,}, ratio 1:{n_ina/n_act:.2f})  '
          f'val={len(y_val):,}  test={len(y_test):,}' + ('  [graphs skipped -- classical models only]' if skip_graphs else ''))
