"""V4 -- SMOTE oversampling, on both caspase-3/7 datasets.

Reviewer 1 (#13): "The caspase-3/7 models didn't seem to perform well with
low F1 scores. Have the authors tried other data balancing strategies, e.g.,
up-sampling or SMOTE?"

Method: imblearn.over_sampling.SMOTE on the raw (natural-imbalance) train
pool's ECFP4 fingerprints, oversampling the active class up to the inactive
count. Matches the sampler already used in the DILI TX2017 replication
('rose' = SMOTE there), applied here to the actual assay Reviewer 1 asked
about, not just the unrelated DILI workflow.

IMPORTANT SCOPE LIMIT: SMOTE interpolates between k-NN fingerprint vectors
to synthesize new points -- there is no defined way to turn a synthetic,
non-integer fingerprint back into a valid molecular graph, so this variant
has no matching graphs_train.pt and applies to the 3 classical models
(RF/SVM/XGBoost) ONLY. This is a real methodological boundary, not an
oversight -- state it plainly in the response rather than working around it.

Usage: python v4_smote_oversample.py
Outputs: ./data/<dataset>/v4_smote/{X,y}_{train,val,test}.npy  (no graphs_*.pt)
"""
import os
import numpy as np
from imblearn.over_sampling import SMOTE

from common import BaseSplit, save_variant, CASPASE_DATASETS, SEED, DATA_ROOT

for name, cfg in CASPASE_DATASETS.items():
    print(f"=== {cfg['label']} ===")
    base = BaseSplit(cfg['cleaned_csv'])

    X_tr_raw = base.X_fp[base.idx_tr_raw]
    y_tr_raw = base.y_all[base.idx_tr_raw]
    n_act_before = int((y_tr_raw == 1).sum())

    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_tr_raw, y_tr_raw)
    n_act_after = int((y_res == 1).sum())
    print(f'  SMOTE: active count {n_act_before:,} -> {n_act_after:,} (synthetic only, inactive count unchanged)')

    # save_variant expects an index array into base.X_fp/base.y_all for the
    # real compounds; here ALL of X_res/y_res is already the final resampled
    # array (real + synthetic), so pass an empty real-index selection and
    # hand the whole resampled array in as "extra".
    out_dir = os.path.join(DATA_ROOT, name, 'v4_smote')
    save_variant(base, np.array([], dtype=int), out_dir,
                 extra_X_train=X_res, extra_y_train=y_res, skip_graphs=True)

print('\nDone: V4 (SMOTE) built for both datasets -- classical models only.')
