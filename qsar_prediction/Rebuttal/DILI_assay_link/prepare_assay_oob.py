"""STEP A -- repeated random splits of the two HepG2 assays, arranged so every
compound can receive an out-of-bag (OOB) in vitro prediction. Also featurizes the
full DILI universe so the assay models can score it.

WHY OUT-OF-BAG AND NOT EXCLUSION
--------------------------------
The linking experiment feeds P(MMP active) and P(caspase-3/7 active) into the DILI
models as extra features. For that to mean anything, a compound's in vitro value
must come from an assay model that never trained on that compound -- otherwise the
value is a memorised label, sharper than anything achievable for a new drug.

The obvious fix, excluding DILI compounds from assay training, does not survive
contact with the data. The DILI models train on the clinical pool (Alloriginal
minus the gold sets, 12,145 compounds), not just the 1,098 gold compounds, so the
exclusion list is the full DILI universe of 13,243 InChIKeys. That overlaps 76% of
MMP (4,513 of 5,937) and 69% of caspase (5,039 of 7,285). Excluding it would leave
1,424 MMP compounds with 205 actives -- the assay models would be worthless and
the features meaningless.

So instead: train on repeated random splits and take each compound's feature from
the repeats in which it was NOT in the training set. Nothing is discarded, and
every feature is still out-of-sample. This is out-of-bag estimation, and it keeps
the whole experiment inside the repeated-random-split scheme used elsewhere in
this project rather than introducing a second protocol.

COVERAGE
--------
With an 80/10/10 split, one repeat holds out ~20% of compounds, so R repeats leave
0.8^R of them never held out: 32.8% at R=5 (measured: 32.2% on the real splits),
1.2% at R=20. R_BASE=20 therefore covers ~99%, and the remainder is closed by
"patch" repeats whose held-out portion is seeded with the still-uncovered
compounds and filled at random. A patch repeat is an ordinary random split in
every other respect -- its model simply trains on an 80% that provably excludes
those compounds. The job asserts 100% coverage before it exits.

Usage: python prepare_assay_oob.py <mmp|caspase|dili>
    mmp / caspase -- that assay's splits and features
    dili          -- the shared DILI compound feature set (prediction targets)

Outputs (./data_<assay>/):
    X_all.npy, y_all.npy, ikeys_all.txt       -- one row per compound
    graphs_all.pt, graphs_all_bond.pt         -- atom graphs (+pe, spd, edgepath), line graphs
    repeat_<r>/idx_{train,val,test}.npy       -- indices into the above
    oob_mask.npy                              -- [R, n_compounds] True = held out that repeat
    split_report.csv                          -- per-repeat sizes, patch flag
Outputs (./data_dili_features/):
    X_dili.npy, graphs_dili.pt, graphs_dili_bond.pt, ikeys_dili.txt, dili_labels.csv
"""
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from features import (canonicalize, smiles_to_inchikey, smiles_to_ecfp4,
                      mol_to_graph, atom_graph_to_bond_graph, compute_laplacian_pe)
from graphormer_true import precompute_structural_encodings
from paths import describe, resolve

SEED = 42
R_BASE = 20            # 0.8^20 ~ 1.2% uncovered, then patched to 0
MAX_PATCH = 5
TEST_FRAC = 0.10
np.random.seed(SEED)
torch.manual_seed(SEED)

ASSAYS = {
    'mmp': dict(csv='AID_720637_cleaned.csv', label='MMP (AID 720637, HepG2)'),
    'caspase': dict(csv='AID_654_1347034_cleaned.csv',
                    label='Caspase-3/7 (AID 654+1347034, HepG2)'),
}

# The DILI universe is the clinical pool AND the gold sets -- the DILI models
# train on the former, so both need in vitro features.
DILI_FILES = {
    'Alloriginal': ('Alloriginal_deduped.csv', 'inchikey'),
    'Gold-1111': ('Gold-1111_deduped.csv', 'InChIKey'),
    'NCV-888': ('NCV-888_deduped.csv', 'InChIKey'),
    'Held-223': ('Held-223_deduped.csv', 'InChIKey'),
}


def load_dili_universe():
    """One row per DILI compound, tagged with the most specific set it belongs to."""
    rank = {'Held-223': 0, 'NCV-888': 1, 'Gold-1111': 2, 'Alloriginal': 3}
    frames = []
    for name, (fname, keycol) in DILI_FILES.items():
        d = pd.read_csv(resolve(fname), usecols=[keycol, 'TOXICITY', 'smiles_r'])
        d = d.rename(columns={keycol: 'InChIKey'}).dropna(subset=['InChIKey'])
        d = d.drop_duplicates('InChIKey')
        d['dili_set'] = name
        d['r'] = rank[name]
        frames.append(d)
        print(f'  {name:12s}: {len(d):6d} unique InChIKeys')
    allf = pd.concat(frames, ignore_index=True).sort_values('r')
    allf = allf.drop_duplicates('InChIKey').drop(columns='r').reset_index(drop=True)
    print(f'  DILI universe: {len(allf)} unique InChIKeys')
    return allf


def dedup_by_inchikey_majority(df, key='ikey', label='label'):
    before, out, dropped = len(df), [], 0
    for _, g in df.groupby(key, sort=False):
        if len(g) == 1:
            out.append(g.iloc[0]); continue
        mean = g[label].mean()
        if mean == 0.5:
            dropped += 1; continue
        row = g.iloc[0].copy(); row[label] = int(mean > 0.5); out.append(row)
    res = pd.DataFrame(out).reset_index(drop=True)
    print(f'  InChIKey dedup: {before} -> {len(res)} '
          f'({before - len(res) - dropped} merged, {dropped} dropped on tie)')
    return res


def attach_graph_extras(g):
    """GraphGPS positional encoding + Graphormer structural encodings.

    Narrow dtypes on purpose: spd holds bucket indices (0..22) so int16 is ample,
    and the edge-path means feed an attention bias where float16 is plenty.
    Together they roughly halve the O(N^2) on-disk cost. The model widens them.
    """
    g.pe = compute_laplacian_pe(g)
    spd, ep = precompute_structural_encodings(g)
    g.spd_flat, g.edgepath_flat = spd.to(torch.int16), ep.to(torch.float16)
    return g


def featurize(smis, labels, what):
    X, graphs, bonds, keep = [], [], [], []
    for i, (smi, lbl) in enumerate(zip(smis, labels)):
        fp = smiles_to_ecfp4(smi)
        g = mol_to_graph(smi, int(lbl))
        if fp is None or g is None or g.x.shape[0] == 0:
            continue
        X.append(fp); graphs.append(attach_graph_extras(g))
        bonds.append(atom_graph_to_bond_graph(g)); keep.append(i)
        if len(keep) % 2000 == 0:
            print(f'    {len(keep)} featurized...', flush=True)
    print(f'  {what}: featurized {len(keep)} / {len(smis)}')
    return np.vstack(X), graphs, bonds, np.array(keep)


def make_splits(y_all, ikeys):
    """R_BASE random 80/10/10 splits, then patch repeats until every compound has
    been held out at least once. Returns (list of index dicts, oob_mask, report)."""
    n = len(y_all)
    idx_all = np.arange(n)
    splits, report = [], []

    def one_split(seed, forced_test=None):
        """Stratified 80/10/10. forced_test is placed in the held-out portion."""
        if forced_test is None or len(forced_test) == 0:
            pool, test = train_test_split(idx_all, test_size=TEST_FRAC,
                                          stratify=y_all, random_state=seed)
        else:
            rest = np.setdiff1d(idx_all, forced_test)
            n_extra = max(int(round(TEST_FRAC * n)) - len(forced_test), 0)
            rng = np.random.RandomState(seed)
            extra = rng.choice(rest, size=min(n_extra, len(rest)), replace=False)
            test = np.concatenate([forced_test, extra])
            pool = np.setdiff1d(idx_all, test)
        y_pool = y_all[pool]
        tr_sub, val_sub = train_test_split(np.arange(len(pool)), test_size=1 / 9,
                                           stratify=y_pool, random_state=seed)
        return pool[tr_sub], pool[val_sub], test

    for r in range(R_BASE):
        tr, va, te = one_split(SEED + r)
        splits.append(dict(train=tr, val=va, test=te))
        report.append(dict(repeat=r, patch=False, n_train_raw=len(tr),
                           n_val=len(va), n_test=len(te)))

    def covered(sp):
        m = np.zeros(n, dtype=bool)
        for s in sp:
            m[s['val']] = True
            m[s['test']] = True
        return m

    for p in range(MAX_PATCH):
        unc = np.where(~covered(splits))[0]
        if len(unc) == 0:
            break
        take = unc[:int(round(TEST_FRAC * n))]
        tr, va, te = one_split(SEED + 1000 + p, forced_test=take)
        splits.append(dict(train=tr, val=va, test=te))
        report.append(dict(repeat=len(splits) - 1, patch=True, n_train_raw=len(tr),
                           n_val=len(va), n_test=len(te)))
        print(f'  patch repeat {len(splits) - 1}: forced {len(take)} '
              f'uncovered compounds into the held-out portion '
              f'({len(unc)} were uncovered)')

    # OOB mask: True where the compound was NOT in that repeat's training set
    R = len(splits)
    oob = np.zeros((R, n), dtype=bool)
    for r, s in enumerate(splits):
        oob[r, s['val']] = True
        oob[r, s['test']] = True
    n_unc = int((~oob.any(axis=0)).sum())
    print(f'  {R} repeats total; mean OOB repeats per compound '
          f'{oob.sum(axis=0).mean():.2f}; uncovered {n_unc}')
    assert n_unc == 0, f'{n_unc} compounds never held out -- raise R_BASE/MAX_PATCH'
    return splits, oob, pd.DataFrame(report)


def upsample_train(tr_idx, y_all, seed):
    """Duplicate actives to parity, per General Response G4."""
    rng = np.random.RandomState(seed)
    act, ina = tr_idx[y_all[tr_idx] == 1], tr_idx[y_all[tr_idx] == 0]
    if len(act) and len(act) < len(ina):
        act = np.concatenate([act, rng.choice(act, len(ina) - len(act), replace=True)])
    out = np.concatenate([act, ina])
    rng.shuffle(out)
    return out


def build_dili_features():
    out_dir = os.path.join(HERE, 'data_dili_features')
    os.makedirs(out_dir, exist_ok=True)
    print('=== DILI compound feature set (prediction targets) ===')
    print('Inputs:')
    describe()
    df = load_dili_universe()
    df['csmi'] = df['smiles_r'].apply(canonicalize)
    df = df.dropna(subset=['csmi']).reset_index(drop=True)
    df['ikey2'] = df['csmi'].apply(smiles_to_inchikey)
    X, graphs, bonds, keep = featurize(df['csmi'].values, df['TOXICITY'].values, 'DILI')
    df = df.iloc[keep].reset_index(drop=True)

    np.save(f'{out_dir}/X_dili.npy', X)
    torch.save(graphs, f'{out_dir}/graphs_dili.pt')
    torch.save(bonds, f'{out_dir}/graphs_dili_bond.pt')
    with open(f'{out_dir}/ikeys_dili.txt', 'w') as fh:
        fh.write('\n'.join(df['InChIKey'].tolist()))
    df[['InChIKey', 'csmi', 'TOXICITY', 'dili_set']].to_csv(
        f'{out_dir}/dili_labels.csv', index=False)
    print(f'  saved {len(df)} DILI compounds -> {out_dir}')
    print(df['dili_set'].value_counts().to_string())


def main_assay(key):
    cfg = ASSAYS[key]
    out_dir = os.path.join(HERE, f'data_{key}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'=== {cfg["label"]} ===')
    print('Inputs:')
    describe()
    print('NOTE: no compounds are excluded. Leakage is handled by taking each')
    print('      in vitro value from repeats where the compound was held out.\n')

    df = pd.read_csv(resolve(cfg['csv']), usecols=[
        'PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME'])
    df['csmi'] = df['PUBCHEM_EXT_DATASOURCE_SMILES'].apply(canonicalize)
    df = df.dropna(subset=['csmi'])
    df = df[df['PUBCHEM_ACTIVITY_OUTCOME'].isin(['Active', 'Inactive'])].copy()
    df['label'] = (df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active').astype(int)
    print(f'  Active/Inactive: {len(df)} '
          f'(act={int(df.label.sum())}, ina={int((df.label == 0).sum())})')
    df['ikey'] = df['csmi'].apply(smiles_to_inchikey)
    df = df.dropna(subset=['ikey']).reset_index(drop=True)
    df = dedup_by_inchikey_majority(df)

    print('\nFeaturizing:')
    X_fp, graphs, bonds, keep = featurize(df['csmi'].values, df['label'].values, 'assay')
    df = df.iloc[keep].reset_index(drop=True)
    y_all, ikeys = df['label'].values, df['ikey'].values
    assert pd.Series(ikeys).is_unique, 'duplicate InChIKeys survived dedup'

    np.save(f'{out_dir}/X_all.npy', X_fp)
    np.save(f'{out_dir}/y_all.npy', y_all)
    torch.save(graphs, f'{out_dir}/graphs_all.pt')
    torch.save(bonds, f'{out_dir}/graphs_all_bond.pt')
    with open(f'{out_dir}/ikeys_all.txt', 'w') as fh:
        fh.write('\n'.join(ikeys))

    print(f'\nBuilding {R_BASE} random 80/10/10 repeats (+ patches for full OOB coverage):')
    splits, oob, report = make_splits(y_all, ikeys)
    np.save(f'{out_dir}/oob_mask.npy', oob)

    for r, s in enumerate(splits):
        rep_dir = f'{out_dir}/repeat_{r}'
        os.makedirs(rep_dir, exist_ok=True)
        tr_up = upsample_train(s['train'], y_all, SEED + r)
        np.save(f'{rep_dir}/idx_train.npy', tr_up)
        np.save(f'{rep_dir}/idx_val.npy', s['val'])
        np.save(f'{rep_dir}/idx_test.npy', s['test'])
        assert not (set(ikeys[tr_up]) & set(ikeys[s['val']])), f'repeat {r}: train/val overlap'
        assert not (set(ikeys[tr_up]) & set(ikeys[s['test']])), f'repeat {r}: train/test overlap'
        report.loc[report.repeat == r, 'n_train_upsampled'] = len(tr_up)
    report.to_csv(f'{out_dir}/split_report.csv', index=False)
    print(f'  wrote {len(splits)} repeats -> {out_dir}')
    print('  ASSERTIONS PASSED: full OOB coverage, no train/val/test overlap')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in list(ASSAYS) + ['dili']:
        print(f'Usage: python {os.path.basename(__file__)} <mmp|caspase|dili>')
        sys.exit(1)
    if sys.argv[1] == 'dili':
        build_dili_features()
    else:
        main_assay(sys.argv[1])
