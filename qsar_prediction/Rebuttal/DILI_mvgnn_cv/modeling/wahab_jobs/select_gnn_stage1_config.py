"""Picks the winning hyperparameter config for ONE GNN architecture from its
3 Stage-1 search results (train_gnn_cv_config.py), by highest mean 5-fold CV
val AUC. Same role as select_stage1_config.py for MV-GNN.

Usage: python select_gnn_stage1_config.py <GCN|GAT|GIN|SAGE>
Outputs: ./results/gnn_cv_search/<arch>/winning_config.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from gnn_model import GNN_CONFIGS

ARCHS = ('GCN', 'GAT', 'GIN', 'SAGE')


def main(arch):
    search_dir = os.path.join(os.path.dirname(__file__), 'results', 'gnn_cv_search', arch.lower())
    n_cfg = len(GNN_CONFIGS[arch])

    config_results = []
    for i in range(n_cfg):
        path = os.path.join(search_dir, f'config_{i}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {path} -- run train_gnn_cv_config.py {arch} {i} first.')
        npz = np.load(path, allow_pickle=True)
        config_results.append({
            'cfg': npz['cfg'].item(),
            'mean_cv_auc': float(npz['mean_cv_auc']),
            'std_cv_auc': float(npz['std_cv_auc']),
        })

    rows = [{'ConfigIdx': i, **r['cfg'], 'MeanCV_ValAUC': r['mean_cv_auc'], 'StdCV_ValAUC': r['std_cv_auc']}
            for i, r in enumerate(config_results)]
    df = pd.DataFrame(rows).sort_values('MeanCV_ValAUC', ascending=False)
    df.to_csv(os.path.join(search_dir, 'hyperparameter_search.csv'), index=False)
    print(f'== {arch} Stage-1 hyperparameter search (sorted by mean CV val AUC) ==')
    print(df.to_string(index=False))

    best_idx = int(df.iloc[0]['ConfigIdx'])
    best_cfg = config_results[best_idx]['cfg']
    print(f"\n>> WINNING CONFIG (idx {best_idx}): {best_cfg}  "
          f"(mean CV val AUC = {config_results[best_idx]['mean_cv_auc']:.4f} "
          f"+/- {config_results[best_idx]['std_cv_auc']:.4f})")

    with open(os.path.join(search_dir, 'winning_config.json'), 'w') as f:
        json.dump({'config_idx': best_idx, 'config': best_cfg}, f, indent=2)
    print(f'\nSaved -> {search_dir}/hyperparameter_search.csv, winning_config.json')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ARCHS:
        print(f'Usage: python {sys.argv[0]} <{"|".join(ARCHS)}>')
        sys.exit(1)
    main(sys.argv[1])
