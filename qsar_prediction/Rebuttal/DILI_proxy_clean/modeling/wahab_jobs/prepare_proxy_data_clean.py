"""Builds the DILI clin_train / clin_val / NCV-888 / held-out-223 split with the
InChIKey-based dedup/exclusion fix, replacing the csmi-only method used in
prepare_proxy_data.py (DILI_mvgnn_proxy/) and in proxy_to_dili_consistent
copy 2/4.ipynb.

Confirmed leakage under the old csmi-only method (Checking_cid_and_inichikey.ipynb):
  - Alloriginal itself is 34% duplicate rows by real molecular identity
    (19,911 raw rows -> 13,123 unique InChIKeys).
  - ~25% of held-out-223 (55/222) had a spelling-variant duplicate still sitting
    in the "clinical" training pool after only excluding Gold-1111 by csmi.
  - ~20% of NCV-888 (174/875) was already present in the training pool under a
    different spelling, causing double-weighting once merged in for Stage-2.

Fix applied here: dedup EVERY file by InChIKey (not csmi), and exclude
Gold-1111/Held-223/NCV-888 from Alloriginal by InChIKey (not csmi). Where a
duplicate InChIKey group disagrees on the TOXICITY label, resolve by majority
vote; drop the compound entirely on an exact tie (ambiguous, no safe default) --
same duplicate-resolution principle as the summary-assay pipeline uses for
replicate CID measurements (average continuous scores / majority-vote outcome),
adapted for DILI's binary label. This directly addresses Reviewer 1 (#4) /
Reviewer 3's comment on justifying the duplicate-resolution procedure.

Builds, for every split: ECFP4 fingerprints (classical models), atom-oriented +
bond-oriented graphs (MV-GNN), and atom-graphs + Laplacian PE (GNN family /
Graphormer / GraphGPS) -- all using the same 23-dim atom / 6-dim bond feature
scheme already used throughout this project's DILI pipelines.

Train/val split: 90/10 of the FULL cleaned clinical pool -- no discarded 10%.
The earlier 80/10/[10 discarded] protocol (inherited from
DILI_mvgnn/prepare_dili_data.py and copy 2.ipynb) threw away a chunk of real,
clean, labeled compounds for no methodological reason; the discarded portion
was never a second test set (held-out-223 is the only test set, and it's
excluded from the clinical pool entirely, upstream of this split) -- it was
just unused. clin_val's only job is Optuna trial selection + early stopping,
which the 10% val split already covers, so the other 10% is folded into
clin_train here instead of being thrown away.

Usage: python prepare_proxy_data_clean.py
Outputs (./data/):
    X_<split>.npy, y_<split>.npy                        -- ECFP4 (classical)
    atom_<split>.pt, bond_<split>.pt                     -- MV-GNN graph pairs
    gt_<split>.pt                                        -- GNN/GT graphs w/ Laplacian PE
  for split in {train, val, ncv, test}
"""
import os
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, inchi
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj

RDLogger.DisableLog('rdApp.*')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

LIVER_DIR = os.path.join(os.path.dirname(__file__), 'dili_data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUT_DIR, exist_ok=True)

N_BITS = 1024
PE_DIM = 8


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


# ── atom / bond features -- identical to DILI_mvgnn/prepare_dili_data.py ──────
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
    """Collapses each InChIKey group to one row. If the group agrees on the
    label, keep it. If it disagrees, resolve by majority vote; on an exact tie
    (ambiguous, no safe default), drop the compound entirely."""
    kept_rows = []
    n_conflict_resolved, n_conflict_dropped = 0, 0
    for ikey, g in df.groupby('inchikey'):
        if len(g) == 1 or g[label_col].nunique() == 1:
            kept_rows.append(g.iloc[0])
            continue
        counts = g[label_col].value_counts()
        if len(counts) == 2 and counts.iloc[0] == counts.iloc[1]:
            n_conflict_dropped += 1
            continue  # exact tie -- ambiguous, drop
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


def save_split(name, X, y, atom_g, bond_g, gt_g):
    np.save(os.path.join(OUT_DIR, f'X_{name}.npy'), X)
    np.save(os.path.join(OUT_DIR, f'y_{name}.npy'), y)
    torch.save(atom_g, os.path.join(OUT_DIR, f'atom_{name}.pt'))
    torch.save(bond_g, os.path.join(OUT_DIR, f'bond_{name}.pt'))
    torch.save(gt_g, os.path.join(OUT_DIR, f'gt_{name}.pt'))


def main():
    print('=== Loading + computing InChIKey for all 4 DILI files ===')
    gold_df = load_and_tag(f'{LIVER_DIR}/DILI_Goldstandard_1111.csv')
    held_df = load_and_tag(f'{LIVER_DIR}/test_data_heldouttest_DILIst_223.csv')
    all_df = load_and_tag(f'{LIVER_DIR}/Alloriginaldata_19911.csv')
    ncv_df = load_and_tag(f'{LIVER_DIR}/test_data_ncv_DILIst_888.csv')

    print('\n=== Deduping each file by InChIKey (majority vote on label conflicts) ===')
    print('Gold-1111:')
    gold_df = dedup_by_inchikey_majority(gold_df)
    print('Held-223:')
    held_df = dedup_by_inchikey_majority(held_df)
    print('Alloriginal:')
    all_df = dedup_by_inchikey_majority(all_df)
    print('NCV-888:')
    ncv_df = dedup_by_inchikey_majority(ncv_df)

    # ── Clinical pool: Alloriginal minus Gold-1111 minus Held-223 minus NCV-888,
    #    ALL by InChIKey (the actual fix -- was csmi-only before) ──────────────
    exclude_ikeys = (set(gold_df['inchikey']) | set(held_df['inchikey'])
                      | set(ncv_df['inchikey']))
    clin_df = all_df[~all_df['inchikey'].isin(exclude_ikeys)].reset_index(drop=True)
    print(f'\nClinical pool (Alloriginal - Gold-1111 - Held-223 - NCV-888, by InChIKey): '
          f'{len(clin_df):,}  DILI+={clin_df.label.sum():,}  DILI-={(clin_df.label == 0).sum():,}')

    # ── Sanity check: zero leakage now (by construction, but verify explicitly) ─
    n_leak_held = clin_df['inchikey'].isin(set(held_df['inchikey'])).sum()
    n_leak_ncv = clin_df['inchikey'].isin(set(ncv_df['inchikey'])).sum()
    assert n_leak_held == 0, f'Held-223 still leaking into clinical pool: {n_leak_held}'
    assert n_leak_ncv == 0, f'NCV-888 still leaking into clinical pool: {n_leak_ncv}'
    print('Leakage check (InChIKey): clinical pool has 0 overlap with Held-223, 0 overlap with NCV-888.')

    # ── 90/10 train/val split -- uses the FULL cleaned clinical pool, no
    #    discarded 10%. The earlier 80/10/[10 discarded] split (inherited from
    #    DILI_mvgnn/prepare_dili_data.py and copy 2.ipynb) threw away a chunk
    #    of real, clean, labeled compounds for no methodological reason -- the
    #    only thing that split's val portion is actually needed for is Optuna
    #    trial selection and early-stopping, which a 10% val set already
    #    covers. The discarded 10% is real training signal; keeping it here
    #    instead of throwing it away. ──────────────────────────────────────
    y_clin = clin_df['label'].values.astype(int)
    idx_c = np.arange(len(y_clin))
    idx_c_tr, idx_c_va = train_test_split(idx_c, test_size=0.10, stratify=y_clin, random_state=SEED)
    clin_train_df = clin_df.iloc[idx_c_tr].reset_index(drop=True)
    clin_val_df = clin_df.iloc[idx_c_va].reset_index(drop=True)
    print(f'Clinical train / val : {len(idx_c_tr):,} / {len(idx_c_va):,}  (90/10, full pool, nothing discarded)')

    print('\n=== Building features (ECFP4 + atom/bond graphs + Laplacian PE) ===')
    X_tr, y_tr, atom_tr, bond_tr, gt_tr = build_all_features(clin_train_df, 'clin_train')
    X_va, y_va, atom_va, bond_va, gt_va = build_all_features(clin_val_df, 'clin_val')
    X_ncv, y_ncv, atom_ncv, bond_ncv, gt_ncv = build_all_features(ncv_df, 'NCV-888')
    X_te, y_te, atom_te, bond_te, gt_te = build_all_features(held_df, 'held_test (223)')

    save_split('train', X_tr, y_tr, atom_tr, bond_tr, gt_tr)
    save_split('val', X_va, y_va, atom_va, bond_va, gt_va)
    save_split('ncv', X_ncv, y_ncv, atom_ncv, bond_ncv, gt_ncv)
    save_split('test', X_te, y_te, atom_te, bond_te, gt_te)

    print(f'\nAtom feature dim: {ATOM_DIM}  Bond feature dim: {BOND_DIM}  ECFP4 bits: {N_BITS}')
    print(f'Saved all split files to {OUT_DIR}/')


if __name__ == '__main__':
    main()
