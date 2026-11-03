"""Builds ECFP4 1024-bit fingerprints for the TX2017-Structure split runs.

Same featurization as ../DILI_tx2017/prepare_features_clean.py (RDKit
canonical SMILES -> Morgan radius 2, 1024 bits), applied to the shared DILI
compound arrays so the split indices in splits/ address the same rows as every
other DILI split-protocol run.

Order matters: the rows are written in the order of data_tx/ikeys_<part>.txt,
which is the order the split indices were built against. The script asserts the
label vector it reconstructs matches the stored y_<part>.npy, so a SMILES or
ordering mistake fails here instead of silently shifting every compound.

Usage: python prepare_tx_features.py
Outputs (data_tx/): X1024_{clin,ncv,test}.npy
"""
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_tx import DATA, N_BITS, read_lines


def fingerprints(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog('rdApp.*')
    out, bad = np.zeros((len(smiles), N_BITS), dtype=np.float32), []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            bad.append(i)
            continue
        # canonicalize first, exactly as the TX2017 prep does
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=N_BITS)
        arr = np.zeros(N_BITS, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return out, bad


def main():
    lab = pd.read_csv(f'{DATA}/dili_labels.csv').drop_duplicates('InChIKey').set_index('InChIKey')
    for part in ('clin', 'ncv', 'test'):
        keys = read_lines(f'{DATA}/ikeys_{part}.txt')
        y_ref = np.load(f'{DATA}/y_{part}.npy').astype(int)
        assert len(keys) == len(y_ref), f'{part}: ikeys and labels disagree'
        sub = lab.loc[keys]
        assert np.array_equal(sub['TOXICITY'].values.astype(int), y_ref), \
            f'{part}: label order does not match y_{part}.npy'
        X, bad = fingerprints(sub['csmi'].values)
        assert not bad, f'{part}: RDKit could not parse rows {bad[:5]}'
        np.save(f'{DATA}/X1024_{part}.npy', X)
        print(f'{part:5s}: {X.shape[0]:6d} compounds x {X.shape[1]} bits, '
              f'DILI+ {100 * y_ref.mean():.1f}%, mean bits set {X.sum(1).mean():.1f}')
    print(f'-> {DATA}')


if __name__ == '__main__':
    main()
