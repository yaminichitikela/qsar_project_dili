"""
Builds 5 repeated random 8:1:1 splits (different seed each repeat), for each
of the three summary assays, mirroring the exact split/feature logic of each
dataset's own prepare_split.py (same RDKit canonicalization, same ECFP4
params, same 1:1 train-only downsampling) -- just repeated 5 times with a
fresh seed per repeat, the same "repeat via different seeds" convention
already used by prepare_scaffold_splits.py for the scaffold-split method.

Only classical-ML (fingerprint) arrays are built here -- this repeated-random-
split experiment is for Method 1 of General Response G3 (random 80/10/10
split), to give it the same repeat-based CI/significance capability that
10-fold CV and the repeated scaffold split already have.

Usage: python prepare_repeats.py
Outputs: ./data/<assay_key>/repeat_<r>/{X,y}_{train,val,test}.npy
"""
import warnings; warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
N_REPEATS = 5
SEED = 42

ASSAYS = {
    'MMP_720637': f'{BASE}/Summary_1_models/modeling/wahab_jobs/cleaned_data/AID_720637_cleaned.csv',
    'Caspase_HepG2_654_1347034': f'{BASE}/summary_2_1437034/modeling/wahab_jobs/cleaned_data/AID_654_1347034_cleaned.csv',
    'Caspase_CHOK1_1347037': f'{BASE}/summary_1347037/modeling/wahab_jobs/cleaned_data/AID_1347037_cleaned.csv',
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


def main():
    for assay_key, cleaned_csv in ASSAYS.items():
        print(f'\n=== {assay_key} ===')
        df_raw = pd.read_csv(cleaned_csv, usecols=[
            'PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES',
            'PUBCHEM_ACTIVITY_SCORE', 'PUBCHEM_ACTIVITY_OUTCOME'])
        df_raw['csmi'] = df_raw['PUBCHEM_EXT_DATASOURCE_SMILES'].apply(canonicalize)
        df = df_raw.dropna(subset=['csmi']).reset_index(drop=True)
        df = df[df['PUBCHEM_ACTIVITY_OUTCOME'].isin(['Active', 'Inactive'])].copy()
        df['label'] = (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)
        print(f'Compounds after cleanup: {len(df):,}  '
              f'(Active={df.label.sum():,}, Inactive={(df.label==0).sum():,})')

        fps = df['csmi'].apply(smiles_to_ecfp4)
        valid = fps.notna()
        df = df[valid].reset_index(drop=True)
        X_fp = np.vstack(fps[valid].values)
        y_all = df['label'].values
        idx_all = np.arange(len(y_all))

        out_dir = os.path.join(HERE, 'data', assay_key)
        os.makedirs(out_dir, exist_ok=True)

        for r in range(N_REPEATS):
            seed = SEED + r
            idx_pool, idx_test = train_test_split(idx_all, test_size=0.10, stratify=y_all, random_state=seed)
            pool_sub = np.arange(len(idx_pool))
            y_pool = y_all[idx_pool]
            tr_sub, val_sub = train_test_split(pool_sub, test_size=1/9, stratify=y_pool, random_state=seed)
            idx_tr_raw = idx_pool[tr_sub]
            idx_val = idx_pool[val_sub]

            X_val, y_val = X_fp[idx_val], y_all[idx_val]
            X_test, y_test = X_fp[idx_test], y_all[idx_test]
            X_tr_raw, y_tr_raw = X_fp[idx_tr_raw], y_all[idx_tr_raw]

            act_idx = np.where(y_tr_raw == 1)[0]
            ina_idx = np.where(y_tr_raw == 0)[0]
            rng = np.random.RandomState(seed)
            sampled_ina = rng.choice(ina_idx, size=len(act_idx), replace=False)
            train_idx = np.concatenate([act_idx, sampled_ina])
            rng.shuffle(train_idx)

            X_train = X_tr_raw[train_idx]
            y_train = y_tr_raw[train_idx]

            rep_dir = os.path.join(out_dir, f'repeat_{r}')
            os.makedirs(rep_dir, exist_ok=True)
            np.save(os.path.join(rep_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(rep_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(rep_dir, 'X_val.npy'), X_val)
            np.save(os.path.join(rep_dir, 'y_val.npy'), y_val)
            np.save(os.path.join(rep_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(rep_dir, 'y_test.npy'), y_test)

            print(f'  Repeat {r} (seed={seed}): train={len(y_train):,} (1:1 balanced)  '
                  f'val={len(y_val):,} (act={(y_val==1).sum()})  '
                  f'test={len(y_test):,} (act={(y_test==1).sum()})')

    print('\nAll repeats built.')


if __name__ == '__main__':
    main()
