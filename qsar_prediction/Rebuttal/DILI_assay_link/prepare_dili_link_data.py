"""STEP D(prep) -- builds the DILI side of the linking experiment: repeated random
splits of the clinical pool, plus the in vitro feature block aligned to them.

MIRRORS THE PUBLISHED DILI PIPELINE
-----------------------------------
The split structure is the same as
DILI_proxy_clean/modeling/wahab_jobs/prepare_proxy_data_clean.py, so the
Structure-only arm is comparable to the published Stage-1/Stage-2 results:

    clinical pool = Alloriginal, minus Gold-1111, Held-223 and NCV-888 (by
                    InChIKey, never SMILES)
    Stage 1       = train on the clinical pool          ("Proxy-only")
    Stage 2       = train on clinical pool + NCV-888     ("Proxy+DILI")
    test          = Held-223, scored once, never used for any selection

Two differences, both deliberate:

  1. Repeated random splits. The clinical pool is re-split 5 times (seeds 42..46)
     into train/val instead of the single fixed 90/10 split, so every DILI number
     carries a confidence interval from repeats. This answers Reviewer 2's comment
     3 and matches the protocol used for all three summary assays. Held-223 stays
     fixed, so the repeats vary what the model learns from, not what it is judged
     on. Five repeats here, not the assay side's 21 -- the 21 exist solely to give
     every assay compound out-of-bag coverage, which the DILI side does not need.

  2. The in vitro feature block. Each compound carries P(MMP active) and
     P(caspase active) from build_invitro_features.py. Only the "Structure +
     in vitro" arm receives them; the Structure-only arm sees fingerprints and
     graphs alone. That single difference is the experiment.

NOTE ON "PROXY"
---------------
In the DILI pipeline "Proxy" already means the weak clinical DILI labels from
Alloriginal -- it has nothing to do with the in vitro assays. The two features
here are therefore called "in vitro" throughout, never "proxy", so the stage
names keep their published meaning.

MISSING FEATURE VALUES
----------------------
A compound with no usable in vitro value (no qualifying repeat, or RDKit could
not featurize it for an assay) is dropped from BOTH arms, not imputed. Imputing
would invent in vitro evidence, and keeping it in one arm only would make the
arms incomparable. The count is reported.

Usage: python prepare_dili_link_data.py
Outputs (./data_dili_link/):
    X_{clin,ncv,test}.npy, y_{clin,ncv,test}.npy
    iv_{clin,ncv,test}.npy                 -- [n, 2] in vitro features
    graphs_{clin,ncv,test}.pt, graphs_{clin,ncv,test}_bond.pt
    ikeys_{clin,ncv,test}.txt
    repeat_<r>/idx_{train,val}.npy         -- indices into the clin arrays
    split_report.csv
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

SEED = 42
N_REPEATS = 5
VAL_FRAC = 0.10
OUT = os.path.join(HERE, 'data_dili_link')
FEAT = os.path.join(HERE, 'data_dili_features')
IV_COLS = ['mmp_prob', 'caspase_prob']


def main():
    os.makedirs(OUT, exist_ok=True)

    # ── the featurized DILI universe, built by prepare_assay_oob.py dili ─────
    labels = pd.read_csv(f'{FEAT}/dili_labels.csv')
    keys = [ln.strip() for ln in open(f'{FEAT}/ikeys_dili.txt') if ln.strip()]
    assert len(labels) == len(keys), 'dili_labels.csv and ikeys_dili.txt disagree'
    labels['pos'] = np.arange(len(labels))
    X_all = np.load(f'{FEAT}/X_dili.npy')
    graphs_all = torch.load(f'{FEAT}/graphs_dili.pt', weights_only=False)
    bonds_all = torch.load(f'{FEAT}/graphs_dili_bond.pt', weights_only=False)
    print(f'DILI universe featurized: {len(labels)}')

    # ── the in vitro features ────────────────────────────────────────────────
    iv_path = os.path.join(HERE, 'results', 'invitro_features.csv')
    if not os.path.exists(iv_path):
        print(f'missing {iv_path} -- run build_invitro_features.py first')
        sys.exit(1)
    iv = pd.read_csv(iv_path)[['InChIKey'] + IV_COLS]
    df = labels.merge(iv, on='InChIKey', how='left')
    usable = df[IV_COLS].notna().all(axis=1)
    print(f'in vitro features available for {int(usable.sum())} / {len(df)} '
          f'({int((~usable).sum())} dropped -- no qualifying repeat)')
    df = df[usable].reset_index(drop=True)

    # ── split by DILI set membership, exactly as the published pipeline ──────
    #  Alloriginal rows are the clinical pool (the tag is already the most
    #  specific set each compound belongs to, so Gold/NCV/Held are not in it)
    parts = {'clin': df[df.dili_set == 'Alloriginal'],
             'ncv': df[df.dili_set == 'NCV-888'],
             'test': df[df.dili_set == 'Held-223']}
    for name, part in parts.items():
        print(f'  {name:5s}: {len(part):6d} compounds, '
              f'DILI+ {100 * part.TOXICITY.mean():.1f}%')

    # leakage guards: the three sets must be disjoint, and the test set must not
    # appear in anything the model trains on
    s = {k: set(v.InChIKey) for k, v in parts.items()}
    assert not (s['clin'] & s['test']), 'Held-223 leaked into the clinical pool'
    assert not (s['ncv'] & s['test']), 'Held-223 leaked into NCV-888'
    assert not (s['clin'] & s['ncv']), 'NCV-888 leaked into the clinical pool'
    print('  ASSERTIONS PASSED: clin / ncv / test are disjoint by InChIKey')

    for name, part in parts.items():
        pos = part['pos'].values
        np.save(f'{OUT}/X_{name}.npy', X_all[pos])
        np.save(f'{OUT}/y_{name}.npy', part['TOXICITY'].values.astype(int))
        np.save(f'{OUT}/iv_{name}.npy', part[IV_COLS].values.astype(np.float32))
        torch.save([graphs_all[i] for i in pos], f'{OUT}/graphs_{name}.pt')
        torch.save([bonds_all[i] for i in pos], f'{OUT}/graphs_{name}_bond.pt')
        with open(f'{OUT}/ikeys_{name}.txt', 'w') as fh:
            fh.write('\n'.join(part['InChIKey'].tolist()))

    # ── 5 repeated random train/val splits of the clinical pool ─────────────
    y_clin = parts['clin']['TOXICITY'].values.astype(int)
    idx_all = np.arange(len(y_clin))
    report = []
    for r in range(N_REPEATS):
        tr, va = train_test_split(idx_all, test_size=VAL_FRAC, stratify=y_clin,
                                  random_state=SEED + r)
        rep = f'{OUT}/repeat_{r}'
        os.makedirs(rep, exist_ok=True)
        np.save(f'{rep}/idx_train.npy', tr)
        np.save(f'{rep}/idx_val.npy', va)
        assert not (set(tr) & set(va)), f'repeat {r}: train/val overlap'
        report.append(dict(repeat=r, n_train=len(tr), n_val=len(va),
                           train_dili_pos=int(y_clin[tr].sum()),
                           val_dili_pos=int(y_clin[va].sum())))
        print(f'  repeat {r}: train={len(tr)} (DILI+ {y_clin[tr].sum()}) '
              f'val={len(va)} (DILI+ {y_clin[va].sum()})')
    pd.DataFrame(report).to_csv(f'{OUT}/split_report.csv', index=False)

    print(f'\nStage 1 trains on the clinical pool; Stage 2 adds NCV-888 '
          f'({len(parts["ncv"])} compounds).')
    print(f'Test set is Held-223 ({len(parts["test"])} compounds), scored once.')
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
