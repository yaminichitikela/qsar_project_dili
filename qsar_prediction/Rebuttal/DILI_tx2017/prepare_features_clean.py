"""Builds ECFP4 (1024-bit) features for the TX2017-Structure replication with
the InChIKey-based dedup/exclusion fix -- same method as
DILI_proxy_clean/prepare_proxy_data_clean.py (dedup every file by InChIKey,
majority-vote on label conflicts, exclude Gold-1111/Held-223/NCV-888 from
Alloriginal by InChIKey, 90/10 train/val split of the full cleaned pool, no
discarded 10%) -- applied here to build ECFP4-only features (no graphs
needed, TX2017 is classical-ML-only).

Two training sets are built for the two-stage protocol:
  Stage-1 ("Proxy-only")  : clin_train alone
  Stage-2 ("Proxy+DILI")  : clin_train + NCV-888 combined
Both stages validate on clin_val and test on the SAME held-out-223 -- the
test set is built once and never touched by either stage's training/search.

Usage: python prepare_features_clean.py
Outputs (./data/):
    X_train.npy, y_train.npy         -- clin_train (Stage-1 training set)
    X_val.npy,   y_val.npy           -- clin_val (both stages' validation set)
    X_ncv.npy,   y_ncv.npy           -- NCV-888 (Stage-2 augmentation)
    X_test.npy,  y_test.npy          -- held-out-223 (fixed test set, both stages)
"""
import os
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, inchi
from sklearn.model_selection import train_test_split

RDLogger.DisableLog('rdApp.*')

SEED = 42
np.random.seed(SEED)

LIVER_DIR = os.path.join(os.path.dirname(__file__), 'dili_data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(OUT_DIR, exist_ok=True)

N_BITS = 1024  # matches Xu et al.'s stated ECFP4 bit length exactly


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


def build_features(df, name):
    labels = df['label'].values.astype(int)
    X_list, y_list = [], []
    for smi, lbl in zip(df['csmi'].values, labels):
        fp = smiles_to_ecfp4(smi)
        if fp is None:
            continue
        X_list.append(fp)
        y_list.append(int(lbl))
    y = np.array(y_list)
    print(f'{name}: {len(y):,}  DILI+={int(y.sum()):,}  DILI-={int((y == 0).sum()):,}')
    return np.vstack(X_list), y


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

    # 90/10 split, full pool, nothing discarded -- same as DILI_proxy_clean
    y_clin = clin_df['label'].values.astype(int)
    idx_c = np.arange(len(y_clin))
    idx_c_tr, idx_c_va = train_test_split(idx_c, test_size=0.10, stratify=y_clin, random_state=SEED)
    clin_train_df = clin_df.iloc[idx_c_tr].reset_index(drop=True)
    clin_val_df = clin_df.iloc[idx_c_va].reset_index(drop=True)
    print(f'Clinical train / val : {len(idx_c_tr):,} / {len(idx_c_va):,}  (90/10, full pool, nothing discarded)')

    print('\n=== Building ECFP4 features ===')
    X_tr, y_tr = build_features(clin_train_df, 'clin_train')
    X_va, y_va = build_features(clin_val_df, 'clin_val')
    X_ncv, y_ncv = build_features(ncv_df, 'NCV-888')
    X_te, y_te = build_features(held_df, 'held_test (223)')

    np.save(os.path.join(OUT_DIR, 'X_train.npy'), X_tr)
    np.save(os.path.join(OUT_DIR, 'y_train.npy'), y_tr)
    np.save(os.path.join(OUT_DIR, 'X_val.npy'), X_va)
    np.save(os.path.join(OUT_DIR, 'y_val.npy'), y_va)
    np.save(os.path.join(OUT_DIR, 'X_ncv.npy'), X_ncv)
    np.save(os.path.join(OUT_DIR, 'y_ncv.npy'), y_ncv)
    np.save(os.path.join(OUT_DIR, 'X_test.npy'), X_te)
    np.save(os.path.join(OUT_DIR, 'y_test.npy'), y_te)

    print(f'\nECFP4 dim: {N_BITS}')
    print(f'Saved all feature files to {OUT_DIR}/')


if __name__ == '__main__':
    main()
