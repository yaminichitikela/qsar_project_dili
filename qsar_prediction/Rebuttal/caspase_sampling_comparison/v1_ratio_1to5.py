"""V1 -- random undersampling at 1:5 (active:inactive) instead of the
current 1:1, on both caspase-3/7 datasets.

Reviewer 2 (p.8 l.50): "...poorer performance when softer (1:5) downsampling
was attempted (in Supplemental)." -- that earlier 1:5 result was produced by
the OLD pre-Rebuttal pipeline (different features, different cleaning), so
it isn't comparable to the current 1:1 baseline. This rebuilds 1:5 with the
CURRENT pipeline (same cleaned CSV, same ECFP4/graph features, same val/test)
so it's a fair, direct comparison against V0.

Same random-choice downsampling logic as prepare_split.py -- only the target
ratio changes (size = 5 * n_active instead of 1 * n_active, capped at the
number of inactives actually available in the raw train pool).

Usage: python v1_ratio_1to5.py
Outputs: ./data/<dataset>/v1_ratio_1to5/{X,y}_{train,val,test}.npy, graphs_*.pt
"""
import os
import numpy as np

from common import BaseSplit, save_variant, CASPASE_DATASETS, SEED, DATA_ROOT

RATIO = 5

for name, cfg in CASPASE_DATASETS.items():
    print(f"=== {cfg['label']} ===")
    base = BaseSplit(cfg['cleaned_csv'])

    y_tr_raw = base.y_all[base.idx_tr_raw]
    act_idx = np.where(y_tr_raw == 1)[0]
    ina_idx = np.where(y_tr_raw == 0)[0]
    rng = np.random.RandomState(SEED)
    target_n_ina = min(RATIO * len(act_idx), len(ina_idx))
    sampled_ina = rng.choice(ina_idx, size=target_n_ina, replace=False)
    keep = np.concatenate([act_idx, sampled_ina])
    rng.shuffle(keep)
    train_idx = base.idx_tr_raw[keep]

    out_dir = os.path.join(DATA_ROOT, name, 'v1_ratio_1to5')
    save_variant(base, train_idx, out_dir)

print('\nDone: V1 (random 1:5) built for both datasets.')
