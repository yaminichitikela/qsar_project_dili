"""Before/after table: the Structure-only gap at the published fingerprint
lengths, and again with the fingerprint matched at 1024 bits.

BEFORE  what the SI tables report. Structure only uses the 1024-bit ECFP4
        specified by Xu et al.; RF/SVM/XGB use the 2048-bit ECFP4 of our
        Methods. The two rows therefore differ in representation.
AFTER   the same RF/SVM/XGB, retrained on the 1024-bit fingerprint, so every
        model sees the same representation.

Reads only existing result files plus results/per_unit_all.csv written by
run_classical_hp.py. Writes results/before_after.csv.

Usage: python make_before_after_table.py
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
CPT = 0.5
PROTOS = ['random_repeat', 'cv', 'scaffold']


def load():
    # Structure only, as reported in the SI (1024 -> top-180, Xu's own SVM)
    so = pd.read_csv(f'{REB}/DILI_tx2017_splits/results_down/tx_per_unit_all.csv')
    so = so[(so.Split == 'test') & (so.CPT == CPT) & (so.Model == 'SVM')]

    # BEFORE: our classical models at 2048 bits, as reported in the SI
    before = pd.read_csv(f'{REB}/DILI_split_protocols/results/per_unit_all.csv')
    before = before[(before.Split == 'test') & (before.CPT == CPT)
                    & (before.Protocol.isin(PROTOS))
                    & (before.Model.isin(['RF', 'SVM', 'XGB']))]

    # AFTER: the same models retrained on the 1024-bit fingerprint
    after = pd.read_csv(f'{HERE}/results/per_unit_all.csv')
    after = after[(after.CPT == CPT) & (after.Arm == 'C_allbits')]
    return so, before, after


def main():
    so, before, after = load()
    rows = []
    for stage, slab in [(1, 'Stage 1'), (2, 'Stage 2'), (None, 'Both stages')]:
        def m(df, **f):
            s = df if stage is None else df[df.Stage == stage]
            for k, v in f.items():
                s = s[s[k] == v]
            return s['ROC-AUC'].mean()

        base = m(so)
        for fam, lab in [('RF', 'Random Forest'), ('XGB', 'XGBoost'), ('SVM', 'SVM')]:
            b, a = m(before, Model=fam), m(after, Model=fam)
            rows.append({'Stage': slab, 'Model': lab,
                         'Structure only (1024)': round(base, 3),
                         'BEFORE: ours @2048': round(b, 3),
                         'gap before': round(b - base, 3),
                         'AFTER: ours @1024': round(a, 3),
                         'gap after': round(a - base, 3)})
    df = pd.DataFrame(rows)
    out = f'{HERE}/results/before_after.csv'
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
