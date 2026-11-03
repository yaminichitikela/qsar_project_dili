"""STEP C -- turns the assay models' predictions into the two in vitro features,
using out-of-bag aggregation so no value is ever in-sample.

THE AGGREGATION RULE
--------------------
For each DILI compound c and each assay A:

  if c is itself in assay A's dataset:
      average A's predictions for c over ONLY the repeats where c was held out
      (val or test), read from oob_mask.npy
  else:
      average over ALL repeats -- no model ever saw c, so every repeat qualifies

Either way, every contributing model was trained without c. That is the whole
point: 76% of MMP and 69% of caspase overlap the DILI universe, so simply
predicting with a model trained on everything would hand the DILI models
memorised assay labels for most of their compounds.

Within a repeat, the per-family predictions are averaged into a consensus first
(the 10-model consensus this project reports), then the consensus values are
averaged across the qualifying repeats. Doing it in that order keeps every repeat
equally weighted regardless of how many families finished.

Writes a provenance column per assay (<assay>_n_oob_repeats) so audit_leakage.py
can verify, from the artifacts alone, that each value used only legal repeats.

Usage: python build_invitro_features.py [--families RF,XGB,...]
    --families restricts the consensus to the families whose probs exist, which
    is how the local pilot runs before the GPU families have been trained.

Outputs (./results/):
    invitro_features.csv   -- one row per DILI compound: the two features,
                              provenance counts, DILI label and set membership
    invitro_summary.csv    -- per-assay coverage and distribution summary
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from optuna_spaces import ALL_FAMILIES

ASSAYS = ['mmp', 'caspase']


def available_families(assay, wanted):
    """Families that have a prob file for at least one repeat."""
    d = os.path.join(HERE, 'results', assay, 'probs')
    if not os.path.isdir(d):
        return []
    out = []
    for f in wanted:
        if any(x.startswith(f'{f}_r') and x.endswith('_dili.npy') for x in os.listdir(d)):
            out.append(f)
    return out


def load_assay_predictions(assay, families, n_dili):
    """Returns (stack [R, n_dili] consensus probs, list of repeats present).

    Missing (family, repeat) files are skipped rather than fatal, so a partially
    finished sweep still produces usable features -- the per-repeat consensus is
    the mean over whichever families are present for that repeat.
    """
    d = os.path.join(HERE, 'results', assay, 'probs')
    rep_dir = os.path.join(HERE, f'data_{assay}')
    n_rep = len([x for x in os.listdir(rep_dir) if x.startswith('repeat_')])

    per_repeat, repeats, used = [], [], {}
    for r in range(n_rep):
        mats = []
        for fam in families:
            p = os.path.join(d, f'{fam}_r{r}_dili.npy')
            if os.path.exists(p):
                v = np.load(p)
                if len(v) != n_dili:
                    raise ValueError(f'{p}: {len(v)} predictions, expected {n_dili}')
                mats.append(v.astype(np.float64))
        if mats:
            per_repeat.append(np.mean(mats, axis=0))
            repeats.append(r)
            used[r] = len(mats)
    if not per_repeat:
        raise FileNotFoundError(f'no {assay} predictions in {d}')
    return np.vstack(per_repeat), repeats, used


def main(wanted_families):
    feat_dir = os.path.join(HERE, 'data_dili_features')
    dili = pd.read_csv(f'{feat_dir}/dili_labels.csv')
    dili_keys = [ln.strip() for ln in open(f'{feat_dir}/ikeys_dili.txt') if ln.strip()]
    assert len(dili) == len(dili_keys), 'dili_labels.csv and ikeys_dili.txt disagree'
    n_dili = len(dili_keys)
    key_pos = {k: i for i, k in enumerate(dili_keys)}
    print(f'DILI compounds to feature: {n_dili}')

    out = dili[['InChIKey', 'TOXICITY', 'dili_set']].copy()
    summary = []

    for assay in ASSAYS:
        fams = available_families(assay, wanted_families)
        if not fams:
            print(f'\n{assay}: no predictions found, skipping')
            out[f'{assay}_prob'] = np.nan
            out[f'{assay}_n_oob_repeats'] = 0
            continue
        print(f'\n{assay}: consensus over {len(fams)} families {fams}')

        preds, repeats, used = load_assay_predictions(assay, fams, n_dili)
        print(f'  repeats with predictions: {len(repeats)} '
              f'(families per repeat: {sorted(set(used.values()))})')

        # which DILI compounds are also assay compounds, and where
        assay_keys = [ln.strip() for ln in
                      open(os.path.join(HERE, f'data_{assay}', 'ikeys_all.txt')) if ln.strip()]
        oob = np.load(os.path.join(HERE, f'data_{assay}', 'oob_mask.npy'))
        assay_pos = {k: i for i, k in enumerate(assay_keys)}

        vals = np.full(n_dili, np.nan)
        counts = np.zeros(n_dili, dtype=int)
        n_overlap = 0
        for k, i in key_pos.items():
            j = assay_pos.get(k)
            if j is None:
                # not an assay compound: every repeat's model is out-of-sample
                rows = np.arange(len(repeats))
            else:
                n_overlap += 1
                legal = oob[:, j]                      # [R_total] over all repeats
                rows = np.array([ri for ri, r in enumerate(repeats) if legal[r]])
            if len(rows):
                vals[i] = preds[rows, i].mean()
                counts[i] = len(rows)

        out[f'{assay}_prob'] = vals
        out[f'{assay}_n_oob_repeats'] = counts
        have = ~np.isnan(vals)
        print(f'  overlapping DILI compounds (OOB-restricted): {n_overlap}')
        print(f'  features produced: {int(have.sum())} / {n_dili}'
              f'  ({int((~have).sum())} without a usable repeat)')
        print(f'  contributing repeats per compound: '
              f'min={counts[have].min()} mean={counts[have].mean():.2f} max={counts[have].max()}')
        print(f'  P(active) distribution: mean={vals[have].mean():.4f} '
              f'median={np.median(vals[have]):.4f} '
              f'[{vals[have].min():.4f}, {vals[have].max():.4f}]')
        summary.append(dict(assay=assay, families=len(fams), repeats=len(repeats),
                            overlap=n_overlap, featured=int(have.sum()),
                            mean_prob=round(float(vals[have].mean()), 4),
                            mean_repeats=round(float(counts[have].mean()), 2)))

    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    out.to_csv(os.path.join(HERE, 'results', 'invitro_features.csv'), index=False)
    pd.DataFrame(summary).to_csv(
        os.path.join(HERE, 'results', 'invitro_summary.csv'), index=False)
    print(f'\n-> results/invitro_features.csv  ({len(out)} rows)')

    # a first look: do the in vitro predictions differ between DILI+ and DILI-?
    print('\nIn vitro prediction by DILI label (descriptive, not the experiment):')
    for assay in ASSAYS:
        col = f'{assay}_prob'
        if out[col].notna().sum() == 0:
            continue
        g = out.dropna(subset=[col]).groupby('TOXICITY')[col].agg(['count', 'mean', 'median'])
        print(f'  {assay}:')
        for tox, row in g.iterrows():
            print(f'    DILI={int(tox)}  n={int(row["count"]):5d}  '
                  f'mean P(active)={row["mean"]:.4f}  median={row["median"]:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', default=','.join(ALL_FAMILIES),
                    help='comma-separated families to include in the consensus')
    a = ap.parse_args()
    main([f.strip() for f in a.families.split(',') if f.strip()])
