"""Combines the per-unit (per-repeat or per-fold) Optuna results for all 6
graph architectures -- produced by train_gnn_optuna.py / train_gt_optuna.py --
into one per-unit CSV and one mean+/-std summary CSV per split type. Also
computes graph-only consensus groups (Consensus-GNN, Consensus-GT,
Consensus-AllGraph) from the saved test-set probabilities, the same simple
unweighted-mean convention as compute_consensus.py / _loo.py / _scaffold.py
in Summary_1_models.

Only graph models are included here (per the point of this folder); combining
with the classical-model grid-search results from Summary_1_models to get a
full 9/10-model consensus is a separate step, not done by this script.

Usage: python aggregate_optuna.py <repeat|loo|scaffold>
Requires all 6 architectures' train_gnn_optuna.py / train_gt_optuna.py runs
to have finished for every unit of the chosen split.

Outputs (./results_<split>/):
    {arch}_{split}_per_unit.csv, {arch}_{split}_summary.csv   (per architecture)
    consensus_{split}_per_unit.csv, consensus_{split}_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS

GNN = ['gcn', 'gat', 'gin', 'sage']
GT = ['graphormer', 'graphgps']
ALL_ARCHS = GNN + GT

CONSENSUS_GROUPS = {
    'Consensus-GNN': GNN,
    'Consensus-GT': GT,
    'Consensus-AllGraph': ALL_ARCHS,
}

SPLITS = {
    'repeat': {'dir': 'data_repeat', 'unit': 'Repeat', 'n_units': 5, 'tag': 'repeat'},
    'loo': {'dir': 'data_loo', 'unit': 'Fold', 'n_units': 10, 'tag': 'loo'},
    'scaffold': {'dir': 'data_scaffold', 'unit': 'Repeat', 'n_units': 5, 'tag': 'scaffold'},
}

HERE = os.path.dirname(__file__)
NUMERIC_COLS = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']


def summarize(df_test, out_dir, name_lower, tag):
    """df_test must have columns METRIC_COLS (Test split only)."""
    df_test.to_csv(os.path.join(out_dir, f'{name_lower}_{tag}_per_{SPLITS_TAG_UNIT[tag]}.csv'), index=False)
    summary_rows = []
    for cpt in [0.5, 0.8]:
        sub = df_test[df_test['CPT'] == cpt]
        mean_row = sub[NUMERIC_COLS].mean().round(4)
        std_row = sub[NUMERIC_COLS].std().round(4)
        summary_rows.append({'Model': name_lower, 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
        summary_rows.append({'Model': name_lower, 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, f'{name_lower}_{tag}_summary.csv'), index=False)


SPLITS_TAG_UNIT = {'repeat': 'repeat', 'loo': 'fold', 'scaffold': 'repeat'}


def main(split):
    cfg = SPLITS[split]
    tag = cfg['tag']
    out_dir = os.path.join(HERE, f'results_{split}')
    probs_dir = os.path.join(out_dir, 'probs')
    data_dir = os.path.join(HERE, cfg['dir'])
    n_units = cfg['n_units']
    unit_dir_prefix = 'repeat' if tag != 'loo' else 'fold'

    # ── per-architecture per-unit + summary ─────────────────────────────────
    for arch in ALL_ARCHS:
        rows = []
        for idx in range(n_units):
            path = os.path.join(out_dir, f'{arch}_{idx}_results.csv')
            if not os.path.exists(path):
                raise FileNotFoundError(f'Missing {path} -- run train_*_optuna.py {arch} {split} {idx} first.')
            rows.append(pd.read_csv(path))
        df_all = pd.concat(rows, ignore_index=True)
        df_test = df_all[df_all['Split'] == 'Test'].copy()
        summarize(df_test, out_dir, arch, tag)
        print(f'{arch}: combined {n_units} units -> {arch}_{tag}_summary.csv')

    # ── graph-only consensus groups, from saved test-set probs ──────────────
    consensus_rows = []
    for idx in range(n_units):
        unit_dir = os.path.join(data_dir, f'{unit_dir_prefix}_{idx}')
        y_test = np.load(os.path.join(unit_dir, 'y_test.npy'))

        probs = {}
        for arch in ALL_ARCHS:
            p = os.path.join(probs_dir, f'{arch}_{idx}_test_probs.npy')
            if not os.path.exists(p):
                raise FileNotFoundError(f'Missing {p}')
            probs[arch] = np.load(p)

        for name, members in CONSENSUS_GROUPS.items():
            consensus_prob = np.mean([probs[a] for a in members], axis=0)
            for cpt in [0.5, 0.8]:
                row = {**evaluate(y_test, consensus_prob, name, cpt), 'Split': 'Test', cfg['unit']: idx}
                consensus_rows.append(row)

    df_cons = pd.DataFrame(consensus_rows)[[cfg['unit'], 'Split'] + METRIC_COLS]
    df_cons.to_csv(os.path.join(out_dir, f'consensus_{tag}_per_{SPLITS_TAG_UNIT[tag]}.csv'), index=False)

    summary_rows = []
    for name in CONSENSUS_GROUPS:
        for cpt in [0.5, 0.8]:
            sub = df_cons[(df_cons['Model'] == name) & (df_cons['CPT'] == cpt)]
            mean_row = sub[NUMERIC_COLS].mean().round(4)
            std_row = sub[NUMERIC_COLS].std().round(4)
            summary_rows.append({'Model': name, 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
            summary_rows.append({'Model': name, 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})
    df_cons_summary = pd.DataFrame(summary_rows)
    df_cons_summary.to_csv(os.path.join(out_dir, f'consensus_{tag}_summary.csv'), index=False)

    print(f'\n== {split}: graph-only consensus mean +/- std across {n_units} units (Test, CPT=0.5) ==')
    print(df_cons_summary[(df_cons_summary['CPT'] == 0.5) & (df_cons_summary['Stat'] == 'mean')])
    print(f'\nSaved -> {out_dir}/consensus_{tag}_summary.csv and per-architecture summaries')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in SPLITS:
        print(f'Usage: python {sys.argv[0]} <repeat|loo|scaffold>')
        sys.exit(1)
    main(sys.argv[1])
