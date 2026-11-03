"""Combines MV-GNN's new Stage-1 ('Proxy-only'/Clinical-only) and Stage-2
('Proxy+DILI'/Clinical+NCV-888) held-out-223 predictions with the existing
RF/XGB/Graphormer/GraphGPS predictions (saved from proxy_to_dili_consistent
copy 2.ipynb, copied into ./probs_baseline/) to reproduce Table 6's
Consensus-Classic/-GT/-Full groups (sanity check) and add a NEW
Consensus-Full+MVGNN group per stage -- without touching/recomputing anything
already published (same "add, don't replace" rule used for Consensus-MVGNN
in the summary-assay pipeline).

Note: ./probs_baseline/p_gps_*.npy are GraphGPS's PLAIN 0.5-threshold
predictions from copy 2.ipynb. The published Table 6's GraphGPS row actually
used a separately re-trained, val-optimised-threshold model from
proxy_to_dili_consistent copy 4.ipynb ("GraphGPS-OptF1"), whose probabilities
were never saved to disk -- so the Consensus-Full number reproduced here may
differ slightly from the exact published Consensus-Full row. This is an
existing inconsistency between copy 2 and copy 4 of proxy_to_dili_consistent,
not something introduced here.

Requires train_mvgnn_stage1.py and train_mvgnn_stage2.py to have run first.
Usage: python compute_consensus_proxy.py
Outputs: ./results/proxy_table6_with_mvgnn.csv
"""
import os

import numpy as np
import pandas as pd

from eval_utils import evaluate, METRIC_COLS

HERE = os.path.dirname(__file__)
BASELINE_DIR = os.path.join(HERE, 'probs_baseline')
RESULTS_DIR = os.path.join(HERE, 'results')
DATA_DIR = os.path.join(HERE, 'data')


def main():
    y_held_baseline = np.load(os.path.join(BASELINE_DIR, 'y_held.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    assert np.array_equal(y_held_baseline, y_test), (
        'held-out-223 label order mismatch between probs_baseline/y_held.npy '
        '(from the notebook) and data/y_test.npy (from prepare_proxy_data.py) '
        '-- do not proceed, probabilities would be misaligned.')
    y = y_test
    print(f'held-out label alignment verified: {len(y)} compounds, {y.sum()} DILI+')

    p_rf_clin = np.load(os.path.join(BASELINE_DIR, 'p_rf_clin.npy'))
    p_rf_aug = np.load(os.path.join(BASELINE_DIR, 'p_rf_aug.npy'))
    p_xgb_clin = np.load(os.path.join(BASELINE_DIR, 'p_xgb_clin.npy'))
    p_xgb_aug = np.load(os.path.join(BASELINE_DIR, 'p_xgb_aug.npy'))
    p_grph_clin = np.load(os.path.join(BASELINE_DIR, 'p_grph_clin.npy'))
    p_grph_aug = np.load(os.path.join(BASELINE_DIR, 'p_grph_aug.npy'))
    p_gps_clin = np.load(os.path.join(BASELINE_DIR, 'p_gps_clin.npy'))
    p_gps_aug = np.load(os.path.join(BASELINE_DIR, 'p_gps_aug.npy'))

    p_mvgnn_clin = np.load(os.path.join(RESULTS_DIR, 'mvgnn_stage1_test_probs.npy'))
    p_mvgnn_aug = np.load(os.path.join(RESULTS_DIR, 'mvgnn_stage2_test_probs.npy'))

    # ── Consensus groups -- Classic/GT/Full reproduce Table 6 as a sanity
    #    check; Full+MVGNN is the new group being added. ────────────────────
    p_cons_classic_clin = np.mean([p_rf_clin, p_xgb_clin], axis=0)
    p_cons_classic_aug = np.mean([p_rf_aug, p_xgb_aug], axis=0)
    p_cons_gt_clin = np.mean([p_grph_clin, p_gps_clin], axis=0)
    p_cons_gt_aug = np.mean([p_grph_aug, p_gps_aug], axis=0)
    p_cons_full_clin = np.mean([p_rf_clin, p_xgb_clin, p_grph_clin, p_gps_clin], axis=0)
    p_cons_full_aug = np.mean([p_rf_aug, p_xgb_aug, p_grph_aug, p_gps_aug], axis=0)
    p_cons_full_mvgnn_clin = np.mean(
        [p_rf_clin, p_xgb_clin, p_grph_clin, p_gps_clin, p_mvgnn_clin], axis=0)
    p_cons_full_mvgnn_aug = np.mean(
        [p_rf_aug, p_xgb_aug, p_grph_aug, p_gps_aug, p_mvgnn_aug], axis=0)

    model_rows = [
        ('RF          (Proxy-only)',            p_rf_clin),
        ('RF          (Proxy+DILI)',            p_rf_aug),
        ('XGB         (Proxy-only)',            p_xgb_clin),
        ('XGB         (Proxy+DILI)',            p_xgb_aug),
        ('Graphormer  (Proxy-only)',            p_grph_clin),
        ('Graphormer  (Proxy+DILI)',            p_grph_aug),
        ('GraphGPS    (Proxy-only)',            p_gps_clin),
        ('GraphGPS    (Proxy+DILI)',            p_gps_aug),
        ('MV-GNN      (Proxy-only)',            p_mvgnn_clin),
        ('MV-GNN      (Proxy+DILI)',            p_mvgnn_aug),
        ('Consensus-Classic (Proxy-only)',      p_cons_classic_clin),
        ('Consensus-Classic (Proxy+DILI)',      p_cons_classic_aug),
        ('Consensus-GT      (Proxy-only)',      p_cons_gt_clin),
        ('Consensus-GT      (Proxy+DILI)',      p_cons_gt_aug),
        ('Consensus         (Proxy-only)',      p_cons_full_clin),
        ('Consensus         (Proxy+DILI)',      p_cons_full_aug),
        ('Consensus-MVGNN   (Proxy-only)',      p_cons_full_mvgnn_clin),
        ('Consensus-MVGNN   (Proxy+DILI)',      p_cons_full_mvgnn_aug),
    ]

    rows = []
    for name, prob in model_rows:
        for cpt in [0.5, 0.8]:
            rows.append(evaluate(y, prob, name.strip(), cpt))

    df = pd.DataFrame(rows)[METRIC_COLS]
    out_path = os.path.join(RESULTS_DIR, 'proxy_table6_with_mvgnn.csv')
    df.to_csv(out_path, index=False)

    print('\n== Table 6 + MV-GNN (held-out-223/222, CPT=0.5 shown) ==')
    print(df[df['CPT'] == 0.5].to_string(index=False))
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
