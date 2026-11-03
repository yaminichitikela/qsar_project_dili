"""V3 -- diversity-based undersampling, 1:1, on both caspase-3/7 datasets.

Reviewer 2 (p.8 l.50): "...a combination of similarity-based downsampling
and/or diversity-based downsampling are more realistic and more stable ways
for balancing out the training data."

Method: MaxMin picking (Ashton et al., QSAR Comb. Sci. 21 (2002), 598-604 --
RDKit's rdSimDivPickers.MaxMinPicker) on the inactive pool's ECFP4
fingerprints, picking n_active inactives that are maximally diverse from
each other (Tanimoto distance). This is the complementary strategy to V2:
V2 asks "which inactives look like actives" (hardest negatives), V3 asks
"which inactives represent the full chemical space of the inactive pool"
(no redundant near-duplicates skewing the training distribution) -- the
reviewer names both as candidates, so both are run and compared rather than
picking one a priori.

Usage: python v3_diversity_downsample.py
Outputs: ./data/<dataset>/v3_diversity/{X,y}_{train,val,test}.npy, graphs_*.pt
"""
import os
import numpy as np
from rdkit.SimDivFilters import rdSimDivPickers

from common import BaseSplit, save_variant, smiles_to_ecfp4_bitvect, CASPASE_DATASETS, SEED, DATA_ROOT

for name, cfg in CASPASE_DATASETS.items():
    print(f"=== {cfg['label']} ===")
    base = BaseSplit(cfg['cleaned_csv'])

    y_tr_raw = base.y_all[base.idx_tr_raw]
    act_pos = np.where(y_tr_raw == 1)[0]
    ina_pos = np.where(y_tr_raw == 0)[0]
    act_global = base.idx_tr_raw[act_pos]
    ina_global = base.idx_tr_raw[ina_pos]
    n_active = len(act_global)

    inactive_fps = [smiles_to_ecfp4_bitvect(s) for s in base.csmi_all[ina_global]]

    picker = rdSimDivPickers.MaxMinPicker()
    picked = list(picker.LazyBitVectorPick(inactive_fps, len(inactive_fps), n_active, seed=SEED))
    keep_ina_global = ina_global[np.array(picked)]

    train_idx = np.concatenate([act_global, keep_ina_global])
    rng = np.random.RandomState(SEED)
    rng.shuffle(train_idx)

    out_dir = os.path.join(DATA_ROOT, name, 'v3_diversity')
    save_variant(base, train_idx, out_dir)

print('\nDone: V3 (diversity-based 1:1, MaxMin) built for both datasets.')
