"""Recomputes the consensus ensembles with the corrected graph transformers.

Three of the five consensus variants contain graph transformers, so correcting
Graphormer and GraphGPS changes them even though nothing else was retrained:

    Consensus-Classical  mean(RF, SVM, XGBoost)                 -- UNCHANGED
    Consensus-GNN        mean(GCN, GAT, GIN, SAGE)              -- UNCHANGED
    Consensus-GT         mean(Graphormer, GraphGPS)             -- fully changed
    Consensus            mean(the 9 above)                      -- changed
    Consensus-MVGNN      mean(the 9 above + MV-GNN-cross)       -- changed

The groupings are transcribed from compute_consensus_loo.py in
Summary_1_models/modeling/wahab_jobs, including the detail that MV-GNN-cross is
excluded from Consensus and appears only in Consensus-MVGNN.

The eight non-transformer families are read from their published per-unit
probability files and not retrained, so the only thing that moves is what the
corrected transformers contribute.

SELF-VALIDATION
---------------
--use-published substitutes the PUBLISHED Graphormer/GraphGPS probabilities for
the new ones. The consensus values it produces must then reproduce the published
consensus numbers. That is a direct check that the groupings, file matching and
averaging here are right, run before trusting any corrected number. Use it on a
protocol whose published consensus is already in the results folder.

Usage:
    python compute_consensus_corrected.py <assay> <protocol> [--use-published]
Outputs (./results/<assay>/<protocol>/):
    probs/Consensus*_<unit>_test_probs.npy
    consensus_<unit>_metrics.csv
    gt_old_vs_new_<protocol>.csv   -- per-unit old/new comparison for the GT rows
"""
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_map import (ASSAY_LABEL, ASSAY_PROTOCOLS, PROTOCOLS, data_dir,
                      out_dir, prob_path, units)
from eval_utils import METRIC_COLS, evaluate

CLASSICAL = ['rf', 'svm', 'xgb']
GNN = ['gcn', 'gat', 'gin', 'sage']
GT = ['graphormer', 'graphgps']
MVGNN = 'mvgnn'


def new_gt_probs(assay, protocol, unit, arch):
    return os.path.join(out_dir(assay, protocol), 'probs',
                        f'{arch}_{unit}_test_probs.npy')


def main(assay, protocol, use_published):
    if protocol not in ASSAY_PROTOCOLS[assay]:
        print(f'{assay} does not report {protocol}')
        sys.exit(1)
    d = out_dir(assay, protocol)
    os.makedirs(os.path.join(d, 'probs'), exist_ok=True)
    print(f'=== {ASSAY_LABEL[assay]} | {protocol}')
    print(f'    {PROTOCOLS[protocol]["caption"]}')
    if use_published:
        print('    MODE: --use-published (validation; must reproduce the '
              'published consensus)')
    has_mv = PROTOCOLS[protocol]['has_mvgnn']

    rows, compare = [], []
    for unit in units(assay, protocol):
        y = np.load(os.path.join(data_dir(assay, protocol, unit), 'y_test.npy')).astype(int)
        probs = {}
        for m in CLASSICAL + GNN + ([MVGNN] if has_mv else []):
            probs[m] = np.load(prob_path(assay, protocol, unit, m))

        for arch, key in (('Graphormer', 'graphormer'), ('GraphGPS', 'graphgps')):
            if use_published:
                probs[key] = np.load(prob_path(assay, protocol, unit, key))
            else:
                p = new_gt_probs(assay, protocol, unit, arch)
                if not os.path.exists(p):
                    print(f'  unit {unit}: missing {p} -- train it first')
                    sys.exit(1)
                probs[key] = np.load(p)

        bad = {k: len(v) for k, v in probs.items() if len(v) != len(y)}
        if bad:
            print(f'  unit {unit}: length mismatch vs y_test ({len(y)}): {bad}')
            sys.exit(1)

        nine = CLASSICAL + GNN + GT
        ens = {
            'Consensus-Classical': np.mean([probs[k] for k in CLASSICAL], axis=0),
            'Consensus-GNN': np.mean([probs[k] for k in GNN], axis=0),
            'Consensus-GT': np.mean([probs[k] for k in GT], axis=0),
            'Consensus': np.mean([probs[k] for k in nine], axis=0),
        }
        if has_mv:
            ens['Consensus-MVGNN'] = np.mean([probs[k] for k in nine + [MVGNN]], axis=0)

        for name, pr in ens.items():
            # validation mode must leave no artifacts behind: its "corrected"
            # probabilities are really the published ones, and a stray file would
            # later be mistaken for a genuine corrected result
            if not use_published:
                np.save(os.path.join(d, 'probs', f'{name}_{unit}_test_probs.npy'),
                        pr.astype(np.float32))
            for cpt in (0.5, 0.8):
                rows.append({**evaluate(y, pr, name, cpt), 'Unit': unit, 'Split': 'Test'})

        # old vs new, for the rows that actually change
        from sklearn.metrics import roc_auc_score
        for key, arch in (('graphormer', 'Graphormer'), ('graphgps', 'GraphGPS')):
            old = np.load(prob_path(assay, protocol, unit, key))
            compare.append(dict(unit=unit, model=arch,
                                old_auc=round(roc_auc_score(y, old), 4),
                                new_auc=round(roc_auc_score(y, probs[key]), 4)))
        old_gt = np.mean([np.load(prob_path(assay, protocol, unit, k)) for k in GT], axis=0)
        old_nine = np.mean([probs[k] if k not in GT else
                            np.load(prob_path(assay, protocol, unit, k))
                            for k in nine], axis=0)
        compare.append(dict(unit=unit, model='Consensus-GT',
                            old_auc=round(roc_auc_score(y, old_gt), 4),
                            new_auc=round(roc_auc_score(y, ens['Consensus-GT']), 4)))
        compare.append(dict(unit=unit, model='Consensus',
                            old_auc=round(roc_auc_score(y, old_nine), 4),
                            new_auc=round(roc_auc_score(y, ens['Consensus']), 4)))
        print(f'  unit {unit}: n_test={len(y)}, {len(ens)} ensembles written')

    cmp_df = pd.DataFrame(compare)
    if not use_published:
        pd.DataFrame(rows)[['Unit', 'Split'] + METRIC_COLS].to_csv(
            os.path.join(d, 'consensus_metrics.csv'), index=False)
        cmp_df.to_csv(os.path.join(d, f'gt_old_vs_new_{protocol}.csv'), index=False)

    print('\nOld (published) vs new (corrected) ROC-AUC, mean over units:')
    g = cmp_df.groupby('model')[['old_auc', 'new_auc']].mean().round(4)
    g['delta'] = (g.new_auc - g.old_auc).round(4)
    print(g.to_string())
    if use_published:
        same = np.allclose(cmp_df.old_auc, cmp_df.new_auc)
        print(f'\nVALIDATION: old == new for every row: {same}')
        if not same:
            print('  MISMATCH -- the grouping or file matching here is wrong')
            sys.exit(1)
    print(f'\n-> {d}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if len(args) != 2 or args[0] not in ASSAY_PROTOCOLS or args[1] not in PROTOCOLS:
        print(f'Usage: python {os.path.basename(__file__)} '
              f'<mmp|hepg2|chok1> <{"|".join(PROTOCOLS)}> [--use-published]')
        sys.exit(1)
    main(args[0], args[1], '--use-published' in sys.argv)
