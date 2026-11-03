"""V2 -- similarity-based undersampling, 1:1, on both caspase-3/7 datasets.

Reviewer 2 (p.8 l.50): "...this could easily result in very dissimilar
inactive datapoints to the opposite (active) class, hence easy overfitting.
Instead a combination of similarity-based downsampling and/or diversity-based
downsampling are more realistic and more stable ways for balancing out the
training data."

Method: for every inactive compound in the raw train pool, compute its
Tanimoto similarity (ECFP4) to every active compound in the same pool, take
the max (its similarity to the *closest* active). Keep the n_active inactives
with the HIGHEST such similarity -- i.e. the inactives most easily confusable
with an active, which is what makes the classification task genuinely hard
rather than artificially easy, directly answering the reviewer's stated
concern. Ratio stays 1:1 so this isolates the effect of *which* inactives
are kept, independent of the ratio question (that's V1).

Usage: python v2_similarity_downsample.py
Outputs: ./data/<dataset>/v2_similarity/{X,y}_{train,val,test}.npy, graphs_*.pt
"""
import os
import numpy as np
from rdkit import DataStructs

from common import BaseSplit, save_variant, smiles_to_ecfp4_bitvect, CASPASE_DATASETS, DATA_ROOT

for name, cfg in CASPASE_DATASETS.items():
    print(f"=== {cfg['label']} ===")
    base = BaseSplit(cfg['cleaned_csv'])

    y_tr_raw = base.y_all[base.idx_tr_raw]
    act_pos = np.where(y_tr_raw == 1)[0]
    ina_pos = np.where(y_tr_raw == 0)[0]
    act_global = base.idx_tr_raw[act_pos]
    ina_global = base.idx_tr_raw[ina_pos]

    active_fps = [smiles_to_ecfp4_bitvect(s) for s in base.csmi_all[act_global]]
    inactive_fps = [smiles_to_ecfp4_bitvect(s) for s in base.csmi_all[ina_global]]

    max_sim_to_active = np.array([
        max(DataStructs.BulkTanimotoSimilarity(fp, active_fps)) for fp in inactive_fps
    ])

    n_active = len(act_global)
    keep_ina_rank = np.argsort(-max_sim_to_active)[:n_active]  # highest similarity first
    keep_ina_global = ina_global[keep_ina_rank]

    print(f'  inactive max-similarity-to-active: kept range '
          f'[{max_sim_to_active[keep_ina_rank].min():.3f}, {max_sim_to_active[keep_ina_rank].max():.3f}]  '
          f'(pool range [{max_sim_to_active.min():.3f}, {max_sim_to_active.max():.3f}])')

    train_idx = np.concatenate([act_global, keep_ina_global])
    rng = np.random.RandomState(42)
    rng.shuffle(train_idx)

    out_dir = os.path.join(DATA_ROOT, name, 'v2_similarity')
    save_variant(base, train_idx, out_dir)

print('\nDone: V2 (similarity-based 1:1) built for both datasets.')
