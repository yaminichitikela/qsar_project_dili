"""Builds the split indices for all four protocols over the DILI clinical pool.

Writes only index files (positions into the clinical-pool arrays), so the output
is a few MB and is built once on the laptop, then rsynced with the folder.

Mirrors the assay split scripts in summary_1347037/modeling/wahab_jobs:
    random_repeat   prepare_random_repeat_splits.py  (10% test, then 1/9 of the
                    remainder as val, stratified, seed 42 + r)
    cv              prepare_loo_folds.py  (StratifiedKFold(10, shuffle, 42);
                    fold i = test, fold (i+1) mod 10 = val, rest = train)
    scaffold        prepare_scaffold_splits.py  (Murcko bins; bins larger than
                    half the test target go to train; remaining bins shuffled
                    with random.Random(42 + r) fill test -> val -> train;
                    not stratified)
Downsampling in all three: inactives sampled without replacement down to the
active count with np.random.RandomState(42 + unit), train only.
random_repeat_nodown reuses random_repeat's splits with the full training set.

The *_up protocols reuse the SAME raw splits and instead balance the training
set by up-sampling: DILI-positive compounds are drawn with replacement up to
the negative count, as in summary_1347037/upsampling (prepare_split_upsample.py,
prepare_loo_folds_upsample.py) and upsampling_protocols. Only the training
indices change -- validation and internal test keep their natural class ratio,
and Held-223 is never part of any split.

Usage: python prepare_splits.py
Outputs: splits/<protocol>/unit_<u>/idx_{train,val,itest}.npy
         splits/split_report.csv
"""
import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BASELINE, PROTOCOLS, SEED, SPLITS, data_dir, read_lines, unit_dir

FRAC_VAL = 0.10
FRAC_TEST = 0.10


def downsample(idx, y, seed):
    """1:1 train-only downsampling, as in the assay scripts."""
    yt = y[idx]
    act = np.where(yt == 1)[0]
    ina = np.where(yt == 0)[0]
    rng = np.random.RandomState(seed)
    if len(ina) >= len(act):
        keep = np.concatenate([act, rng.choice(ina, size=len(act), replace=False)])
    else:                                  # not the case for DILI, kept for safety
        keep = np.concatenate([ina, rng.choice(act, size=len(ina), replace=False)])
    rng.shuffle(keep)
    return idx[keep]


def upsample(idx, y, seed):
    """1:1 train-only UP-sampling, as in the assay up-sampling scripts
    (summary_1347037/upsampling/prepare_split_upsample.py and
    upsampling_protocols/prepare_*_upsample.py): every majority-class compound
    is kept and minority-class compounds are drawn WITH replacement until the
    classes match -- the plain-duplication equivalent of imblearn's
    RandomOverSampler. Applied to the training indices only; the validation and
    internal-test indices are passed through unchanged, and Held-223 is never
    touched by any protocol."""
    yt = y[idx]
    act = np.where(yt == 1)[0]
    ina = np.where(yt == 0)[0]
    rng = np.random.RandomState(seed)
    if len(act) <= len(ina):
        keep = np.concatenate([rng.choice(act, size=len(ina), replace=True), ina])
    else:                                  # not the case for DILI, kept for safety
        keep = np.concatenate([act, rng.choice(ina, size=len(act), replace=True)])
    rng.shuffle(keep)
    return idx[keep]


def murcko(smi):
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(Chem.MolToSmiles(mol))
    except Exception:
        return None


def yang_scaffold_split(scaffolds, n_total, seed):
    """Identical to prepare_scaffold_splits.yang_scaffold_split."""
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
    return (np.array(train_rows), np.array(val_rows), np.array(test_rows),
            len(big_bins))


def save(protocol, unit, tr, va, te, y, report, **extra):
    d = unit_dir(protocol, unit)
    os.makedirs(d, exist_ok=True)
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not (set(a.tolist()) & set(b.tolist())), f'{protocol}/{unit}: overlap'
    if protocol.endswith('_up'):
        # the up-sampled training set must be exactly balanced, and duplication
        # must not have leaked a compound into validation or internal test
        assert int((y[tr] == 1).sum()) == int((y[tr] == 0).sum()), \
            f'{protocol}/{unit}: training set is not 1:1'
    np.save(f'{d}/idx_train.npy', tr.astype(np.int64))
    np.save(f'{d}/idx_val.npy', va.astype(np.int64))
    np.save(f'{d}/idx_itest.npy', te.astype(np.int64))
    report.append(dict(protocol=protocol, unit=unit,
                       n_train=len(tr), train_pos=int(y[tr].sum()),
                       n_val=len(va), val_pos=int(y[va].sum()),
                       n_itest=len(te), itest_pos=int(y[te].sum()),
                       n_train_unique=len(np.unique(tr)), **extra))


def main():
    ddir = data_dir()
    y = np.load(f'{ddir}/y_clin.npy').astype(int)
    keys = read_lines(f'{ddir}/ikeys_clin.txt')
    assert keys == read_lines(f'{BASELINE}/ikeys_clin.txt'), 'clinical pool order changed'
    n = len(y)
    idx_all = np.arange(n)
    print(f'clinical pool: {n} compounds, DILI+ {y.sum()} ({100 * y.mean():.1f}%)')
    report = []

    # ── random repeated 80/10/10 (+ the no-downsampling control) ─────────────
    for r in range(PROTOCOLS['random_repeat']):
        seed = SEED + r
        pool, te = train_test_split(idx_all, test_size=FRAC_TEST, stratify=y,
                                    random_state=seed)
        tr_sub, va_sub = train_test_split(np.arange(len(pool)), test_size=1 / 9,
                                          stratify=y[pool], random_state=seed)
        tr_raw, va = pool[tr_sub], pool[va_sub]
        save('random_repeat', r, downsample(tr_raw, y, seed), va, te, y, report)
        save('random_repeat_nodown', r, tr_raw, va, te, y, report)
        save('random_repeat_up', r, upsample(tr_raw, y, seed), va, te, y, report)

    # ── 10-fold CV ───────────────────────────────────────────────────────────
    k = PROTOCOLS['cv']
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    folds = [t for _, t in skf.split(np.zeros(n), y)]
    for i in range(k):
        te, va = folds[i], folds[(i + 1) % k]
        tr_raw = np.concatenate([folds[j] for j in range(k) if j not in (i, (i + 1) % k)])
        save('cv', i, downsample(tr_raw, y, SEED + i), va, te, y, report)
        save('cv_up', i, upsample(tr_raw, y, SEED + i), va, te, y, report)

    # ── repeated scaffold 8:1:1 ──────────────────────────────────────────────
    lab = pd.read_csv(f'{BASELINE}/dili_labels.csv').drop_duplicates('InChIKey')
    smi = lab.set_index('InChIKey').loc[keys, 'csmi'].values
    scaf = [murcko(s) for s in smi]
    # a molecule RDKit cannot scaffold gets its own bin rather than being dropped,
    # so every clinical compound stays in some split
    scaf = [s if s is not None else f'__noscaf_{i}' for i, s in enumerate(scaf)]
    scaf = [s if s != '' else '__acyclic__' for s in scaf]
    print(f'scaffolds: {len(set(scaf))} bins over {n} compounds')
    for r in range(PROTOCOLS['scaffold']):
        seed = SEED + r
        tr_raw, va, te, n_big = yang_scaffold_split(scaf, n, seed)
        save('scaffold', r, downsample(tr_raw, y, seed), va, te, y, report,
             n_big_bins=n_big)
        save('scaffold_up', r, upsample(tr_raw, y, seed), va, te, y, report,
             n_big_bins=n_big)

    rep = pd.DataFrame(report)
    rep.to_csv(f'{SPLITS}/split_report.csv', index=False)
    print(rep.to_string(index=False))
    print(f'\n-> {SPLITS}')


if __name__ == '__main__':
    main()
