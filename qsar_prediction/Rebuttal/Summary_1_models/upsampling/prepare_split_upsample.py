"""
Same 8:1:1 split as ../modeling/wahab_jobs/prepare_split.py (same SEED, same
stratify columns, val/test are therefore identical to that run), but the
training set is balanced by RANDOM OVERSAMPLING of the active (minority)
class instead of downsampling the inactive (majority) class -- i.e. active
compounds are duplicated with replacement until they match the inactive
count, rather than dropping inactive compounds to match the actives. This
is the plain-duplication equivalent of imblearn's RandomOverSampler, applied
at the index level so it works identically for the fingerprint arrays and
the PyG graph objects.

Usage: python prepare_split_upsample.py
Outputs (written to ./data/):
    X_train.npy, y_train.npy   -- oversampled 1:1 training fingerprints
    X_val.npy,   y_val.npy     -- natural-imbalance validation fingerprints
    X_test.npy,  y_test.npy    -- natural-imbalance test fingerprints
    graphs_train.pt, graphs_val.pt, graphs_test.pt  -- same split, as PyG graphs
"""
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdchem
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

CLEANED_CSV = os.path.join(os.path.dirname(__file__), 'cleaned_data', 'AID_720637_cleaned.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
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


ATOM_SYMBOLS = [6, 7, 8, 16, 9, 15, 17, 35, 53]  # C N O S F P Cl Br I


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
    # ── Step 1: load cleaned data, label ────────────────────────────────────
    df_raw = pd.read_csv(CLEANED_CSV, usecols=[
        'PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES',
        'PUBCHEM_ACTIVITY_SCORE', 'PUBCHEM_ACTIVITY_OUTCOME'])
    df_raw['csmi'] = df_raw['PUBCHEM_EXT_DATASOURCE_SMILES'].apply(canonicalize)
    df = df_raw.dropna(subset=['csmi']).reset_index(drop=True)
    df = df[df['PUBCHEM_ACTIVITY_OUTCOME'].isin(['Active', 'Inactive'])].copy()
    df['label'] = (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)
    print(f'Compounds after cleanup: {len(df):,}  '
          f'(Active={df.label.sum():,}, Inactive={(df.label==0).sum():,})')

    # ── Step 2: features ─────────────────────────────────────────────────────
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
    print(f'Graphs built: {len(graph_list):,}')

    # ── Step 3: 8:1:1 split (identical to prepare_split.py -- same SEED,
    # same stratify columns, so val/test are the same compounds) ────────────
    idx_all = np.arange(len(y_all))
    idx_pool, idx_test = train_test_split(idx_all, test_size=0.10, stratify=y_all, random_state=SEED)
    pool_sub = np.arange(len(idx_pool))
    y_pool = y_all[idx_pool]
    tr_sub, val_sub = train_test_split(pool_sub, test_size=1/9, stratify=y_pool, random_state=SEED)
    idx_tr_raw = idx_pool[tr_sub]
    idx_val = idx_pool[val_sub]

    X_val, y_val = X_fp[idx_val], y_all[idx_val]
    X_test, y_test = X_fp[idx_test], y_all[idx_test]
    X_tr_raw, y_tr_raw = X_fp[idx_tr_raw], y_all[idx_tr_raw]

    # oversample train (actives) to 1:1, instead of downsampling inactives:
    # every inactive is kept, actives are duplicated WITH replacement until
    # the two classes match -- the plain-duplication equivalent of
    # imblearn.over_sampling.RandomOverSampler.
    act_idx = np.where(y_tr_raw == 1)[0]
    ina_idx = np.where(y_tr_raw == 0)[0]
    rng = np.random.RandomState(SEED)
    sampled_act = rng.choice(act_idx, size=len(ina_idx), replace=True)
    train_idx = np.concatenate([sampled_act, ina_idx])
    rng.shuffle(train_idx)

    X_train = X_tr_raw[train_idx]
    y_train = y_tr_raw[train_idx]
    idx_train_final = idx_tr_raw[train_idx]

    print(f'Train (oversampled 1:1): {len(y_train):,}  act={(y_train==1).sum()}, ina={(y_train==0).sum()}  '
          f'(unique actives duplicated: {len(act_idx)} -> {len(sampled_act)})')
    print(f'Val   (natural ratio): {len(y_val):,}  act={(y_val==1).sum()}, ina={(y_val==0).sum()}')
    print(f'Test  (natural ratio): {len(y_test):,}  act={(y_test==1).sum()}, ina={(y_test==0).sum()}')

    # ── save classical ML arrays ─────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_test)

    # ── save graph lists (same split, indexed via valid_g mapping; repeated
    # indices in idx_train_final naturally duplicate the corresponding graph
    # object, which is exactly the oversampling behaviour we want) ──────────
    orig_to_graph_pos = {orig: pos for pos, orig in enumerate(valid_g)}

    def graphs_for(idx_arr):
        return [graph_list[orig_to_graph_pos[i]] for i in idx_arr if i in orig_to_graph_pos]

    torch.save(graphs_for(idx_train_final), os.path.join(OUT_DIR, 'graphs_train.pt'))
    torch.save(graphs_for(idx_val), os.path.join(OUT_DIR, 'graphs_val.pt'))
    torch.save(graphs_for(idx_test), os.path.join(OUT_DIR, 'graphs_test.pt'))

    print(f'\nSaved all split files to {OUT_DIR}/')


if __name__ == '__main__':
    main()
