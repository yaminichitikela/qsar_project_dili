"""Picks the winning hyperparameter config from the 6 Stage-1 search results
(train_mvgnn_cv_config.py jobs 0-5), by highest mean 5-fold CV val AUC --
held-out-223 is never part of this decision (each config's fold_test_probs
were only saved for convenience, not used here).

This replaces select_best_and_ensemble.py's config-selection half; the
ensemble-reporting half is now done downstream by aggregate_stage1_cv.py,
after train_stage1_fold.py retrains the winner once per fold WITH checkpoint
saving (the search-phase configs never save checkpoints, since 5/6 of them
lose and saving 30 checkpoints for one decision would be wasted work).

Requires all 6 train_mvgnn_cv_config.py runs to have finished first (reads
from ./results/cv_search/config_<i>.npz).

Usage: python select_stage1_config.py
Outputs: ./results/stage1_winning_config.json -- {"config_idx": i, "config": {...}}
"""
import json
import os

import numpy as np
import pandas as pd

from cv_configs import CONFIGS

SEARCH_DIR = os.path.join(os.path.dirname(__file__), 'results', 'cv_search')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    config_results = []
    for i in range(len(CONFIGS)):
        path = os.path.join(SEARCH_DIR, f'config_{i}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {path} -- run train_mvgnn_cv_config.py {i} first.')
        npz = np.load(path, allow_pickle=True)
        config_results.append({
            'cfg': npz['cfg'].item(),
            'mean_cv_auc': float(npz['mean_cv_auc']),
            'std_cv_auc': float(npz['std_cv_auc']),
        })

    rows = [{'ConfigIdx': i, **r['cfg'], 'MeanCV_ValAUC': r['mean_cv_auc'], 'StdCV_ValAUC': r['std_cv_auc']}
            for i, r in enumerate(config_results)]
    df = pd.DataFrame(rows).sort_values('MeanCV_ValAUC', ascending=False)
    df.to_csv(os.path.join(OUT_DIR, 'stage1_hyperparameter_search.csv'), index=False)
    print('== Stage-1 hyperparameter search (sorted by mean CV val AUC) ==')
    print(df.to_string(index=False))

    best_idx = int(df.iloc[0]['ConfigIdx'])
    best_cfg = config_results[best_idx]['cfg']
    print(f"\n>> WINNING CONFIG (idx {best_idx}): {best_cfg}  "
          f"(mean CV val AUC = {config_results[best_idx]['mean_cv_auc']:.4f} "
          f"+/- {config_results[best_idx]['std_cv_auc']:.4f})")

    with open(os.path.join(OUT_DIR, 'stage1_winning_config.json'), 'w') as f:
        json.dump({'config_idx': best_idx, 'config': best_cfg}, f, indent=2)
    print(f'\nSaved -> {OUT_DIR}/stage1_hyperparameter_search.csv, stage1_winning_config.json')


if __name__ == '__main__':
    main()
