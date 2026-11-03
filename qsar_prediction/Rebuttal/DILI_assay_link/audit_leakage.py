"""Independent leakage audit for the out-of-bag design.

WHAT THIS CHECKS, AND WHY IT CHANGED
------------------------------------
An earlier version of this audit asked whether DILI compounds had been excluded
from assay training. That was the wrong question. The DILI models train on the
clinical pool (Alloriginal minus the gold sets, 12,145 compounds), not just the
1,098 gold compounds, so the full DILI universe is 13,243 InChIKeys -- 76% of MMP
and 69% of caspase. Excluding it would leave the assay models with 1,424 MMP
compounds and 205 actives, i.e. useless. The old audit passed while silently
ignoring the 12,145 clinical compounds, whose features would have been in-sample.

The design is now out-of-bag, so the property to verify is different:

  for every DILI compound that receives an in vitro value, that value must come
  only from assay repeats in which the compound was NOT in the training set.

This file verifies that from the artifacts on disk, without importing the prep
script, so a bug in the prep cannot hide itself. It checks the OOB mask against
the repeat index files it was supposedly derived from, confirms coverage is
complete, and (once build_invitro_features.py has run) confirms the recorded
provenance of every feature value is consistent with the mask.

Usage: python audit_leakage.py
Exit code is non-zero on failure, so submit_all.sh can gate training on it.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from paths import resolve
ASSAYS = ['mmp', 'caspase']
DILI_FILES = {
    'Alloriginal': ('Alloriginal_deduped.csv', 'inchikey'),
    'Gold-1111': ('Gold-1111_deduped.csv', 'InChIKey'),
    'NCV-888': ('NCV-888_deduped.csv', 'InChIKey'),
    'Held-223': ('Held-223_deduped.csv', 'InChIKey'),
}
KNOWN_INVALID = {'UJNUDOLLRRCQDH-UHFFFAOYSA-N'}   # impossible P valence; RDKit rejects it
fails = []


def check(name, cond, detail=''):
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'  -- {detail}' if detail else ''))
    if not cond:
        fails.append(name)


print('Re-reading the DILI universe from source')
dili_sets = {}
for name, (fname, col) in DILI_FILES.items():
    dili_sets[name] = set(pd.read_csv(resolve(fname), usecols=[col])[col].dropna())
    print(f'  {name:12s} {len(dili_sets[name]):6d}')
dili_all = set().union(*dili_sets.values())
clinical = dili_sets['Alloriginal'] - (dili_sets['Gold-1111'] | dili_sets['NCV-888']
                                       | dili_sets['Held-223'])
print(f'  union {len(dili_all)} (clinical pool {len(clinical)}, '
      f'gold {len(dili_all - clinical)})')

print('\n1. OOB mask must match the repeat index files it claims to describe')
masks, keys = {}, {}
for a in ASSAYS:
    d = f'{HERE}/data_{a}'
    if not os.path.exists(f'{d}/oob_mask.npy'):
        check(f'{a}: oob_mask.npy exists', False, 'run prepare_assay_oob.py first')
        continue
    oob = np.load(f'{d}/oob_mask.npy')
    k = [ln.strip() for ln in open(f'{d}/ikeys_all.txt') if ln.strip()]
    masks[a], keys[a] = oob, k
    R, n = oob.shape
    check(f'{a}: mask shape matches compound count', n == len(k), f'{R} repeats x {n}')
    check(f'{a}: InChIKeys unique', len(k) == len(set(k)))

    rebuilt = np.zeros_like(oob)
    consistent = True
    for r in range(R):
        tr = np.load(f'{d}/repeat_{r}/idx_train.npy')
        va = np.load(f'{d}/repeat_{r}/idx_val.npy')
        te = np.load(f'{d}/repeat_{r}/idx_test.npy')
        rebuilt[r, va] = True
        rebuilt[r, te] = True
        # the decisive property: a compound marked OOB must not be in that
        # repeat's training set, up-sampling duplicates included
        if set(np.unique(tr)) & set(np.where(oob[r])[0]):
            consistent = False
    check(f'{a}: no compound is both OOB and in that repeat\'s training set', consistent)
    check(f'{a}: mask equals val+test membership', np.array_equal(oob, rebuilt))

print('\n2. Every assay compound must be held out at least once')
for a in ASSAYS:
    if a not in masks:
        continue
    cov = masks[a].any(axis=0)
    check(f'{a}: full OOB coverage', cov.all(),
          f'{int((~cov).sum())} uncovered of {len(cov)}; '
          f'mean {masks[a].sum(axis=0).mean():.2f} OOB repeats each')

print('\n3. DILI compounds that overlap an assay must have OOB coverage there')
for a in ASSAYS:
    if a not in masks:
        continue
    kk = np.array(keys[a])
    overlap = np.isin(kk, list(dili_all))
    cov = masks[a].any(axis=0)
    n_ov = int(overlap.sum())
    check(f'{a}: all {n_ov} DILI-overlapping compounds are OOB-covered',
          bool(cov[overlap].all()),
          f'{100 * n_ov / len(kk):.1f}% of the assay overlaps DILI; '
          f'mean {masks[a].sum(axis=0)[overlap].mean():.2f} OOB repeats each')
    # the clinical pool is the part the old audit missed -- check it explicitly
    clin = np.isin(kk, list(clinical))
    check(f'{a}: clinical-pool compounds ({int(clin.sum())}) are OOB-covered',
          bool(cov[clin].all()) if clin.any() else True)

print('\n4. The DILI feature set must cover what the DILI models need')
p = f'{HERE}/data_dili_features/ikeys_dili.txt'
if not os.path.exists(p):
    check('DILI feature set exists', False, 'run prepare_assay_oob.py dili')
else:
    fk = set(ln.strip() for ln in open(p) if ln.strip())
    check('DILI feature set non-empty', len(fk) > 0, f'{len(fk)} compounds')
    missing = dili_sets['Held-223'] - fk
    check('Held-223: only the known-invalid structure is missing',
          missing <= KNOWN_INVALID,
          f'unexpected: {sorted(missing - KNOWN_INVALID)[:3]}' if missing - KNOWN_INVALID
          else f'{len(missing)} missing, as expected')
    check('Held-223 featurized count matches the published DILI test set (222)',
          len(dili_sets['Held-223'] & fk) == 222,
          f'{len(dili_sets["Held-223"] & fk)}')
    for s in ('Gold-1111', 'NCV-888', 'Alloriginal'):
        have = len(dili_sets[s] & fk)
        print(f'        {s:12s} {have}/{len(dili_sets[s])} featurized')
    check('no non-DILI compound in the DILI feature set', len(fk - dili_all) == 0,
          f'{len(fk - dili_all)} stray')

print('\n5. In vitro feature provenance (after build_invitro_features.py)')
prov = f'{HERE}/results/invitro_features.csv'
if not os.path.exists(prov):
    print('  SKIP  build_invitro_features.py has not run yet')
else:
    f = pd.read_csv(prov)
    for a in ASSAYS:
        col, ncol = f'{a}_prob', f'{a}_n_oob_repeats'
        if col not in f or ncol not in f:
            check(f'{a}: provenance columns present', False)
            continue
        have = f[col].notna()
        check(f'{a}: every feature value records >=1 contributing repeat',
              bool((f.loc[have, ncol] >= 1).all()),
              f'{int(have.sum())} compounds with a value')
        check(f'{a}: probabilities in [0, 1]',
              bool(((f.loc[have, col] >= 0) & (f.loc[have, col] <= 1)).all()))
        if a in masks:
            kk = pd.Series(np.arange(len(keys[a])), index=keys[a])
            ov = f[f.InChIKey.isin(kk.index)]
            exp = masks[a].sum(axis=0)[kk.reindex(ov.InChIKey).values]
            check(f'{a}: recorded repeat counts match the OOB mask',
                  bool((ov[ncol].values == exp).all()),
                  f'{len(ov)} overlapping compounds checked')

print('\n' + '=' * 70)
if fails:
    print(f'{len(fails)} CHECK(S) FAILED -- do not start training:')
    for f_ in fails:
        print(f'  - {f_}')
else:
    print('ALL LEAKAGE CHECKS PASSED')
sys.exit(1 if fails else 0)
