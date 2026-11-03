"""Trains ONE model family on ONE split unit of ONE protocol, for one stage.

Same fit procedure as DILI_assay_link/train_dili_link.py `fit` (Structure-only
arm): same trainer, same corrected Graphormer/GraphGPS, same winning
hyperparameters, same epochs and early stopping on validation ROC-AUC, same
Stage-2 warm start. Only the split differs.

Scored on three sets:
    val     the protocol's validation split (used for early stopping)
    itest   the protocol's internal test split from the clinical pool (secondary)
    test    Held-223 (primary; never used for any selection)

Usage:
    python train_protocol.py <protocol> <family> <1|2> <unit>
Env:
    DILI_SMOKE=1   wiring test: 1 epoch, small subsets, writes to results_smoke/

Outputs (results/<protocol>/):
    probs/<family>_s<stage>_u<unit>_{test,itest}.npy
    states/<family>_s1_u<unit>.pt            (graph families, for the warm start)
    <family>_s<stage>_u<unit>_metrics.csv
"""
import json
import os
import sys
import time
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trainer
from common import (ALL_FAMILIES, CLASSICAL, CONFIGS, EPOCHS, PROTOCOLS, RESULTS,
                    check_clin_order, data_dir, unit_dir)
from eval_utils import METRIC_COLS, evaluate

SMOKE = os.environ.get('DILI_SMOKE') == '1'


def load_config(family, stage):
    p = f'{CONFIGS}/{family}_s{stage}_winning_config.json'
    if not os.path.exists(p) and stage == 2 and family not in CLASSICAL:
        # graph families fine-tune Stage 1 and inherit its config, as published
        p = f'{CONFIGS}/{family}_s1_winning_config.json'
    return json.load(open(p)), os.path.basename(p)


def build_data(protocol, family, stage, unit):
    """Stacks clin / ncv / test once and addresses every split by index,
    exactly as train_dili_link.build_split_data does."""
    ddir = data_dir()
    check_clin_order(ddir)
    need_graphs = family not in CLASSICAL
    off, sizes, cursor = {}, {}, 0
    Xs, ys, graphs, bonds = [], [], [], []
    for name in ('clin', 'ncv', 'test'):
        y = np.load(f'{ddir}/y_{name}.npy')
        Xs.append(np.load(f'{ddir}/X_{name}.npy'))
        ys.append(y)
        off[name], sizes[name] = cursor, len(y)
        cursor += len(y)
        if need_graphs:
            import torch
            graphs += torch.load(f'{ddir}/graphs_{name}.pt', weights_only=False)
            if family == 'MVGNN':
                bonds += torch.load(f'{ddir}/graphs_{name}_bond.pt', weights_only=False)

    ud = unit_dir(protocol, unit)
    tr = off['clin'] + np.load(f'{ud}/idx_train.npy')
    va = off['clin'] + np.load(f'{ud}/idx_val.npy')
    it = off['clin'] + np.load(f'{ud}/idx_itest.npy')
    te = off['test'] + np.arange(sizes['test'])
    if stage == 2:
        tr = np.concatenate([tr, off['ncv'] + np.arange(sizes['ncv'])])
    if SMOKE:
        rng = np.random.RandomState(0)
        tr, va, it = (rng.choice(a, min(len(a), m), replace=False)
                      for a, m in ((tr, 400), (va, 200), (it, 200)))

    idx = {'train': tr, 'val': va, 'itest': it, 'test': te}
    data = trainer.SplitData(np.vstack(Xs), np.concatenate(ys), graphs or None,
                             bonds or None, idx, extra=None)
    return data, ddir


def main(protocol, family, stage, unit):
    need_graphs = family not in CLASSICAL
    out = os.path.join(HERE, 'results_smoke' if SMOKE else 'results', protocol)
    os.makedirs(f'{out}/probs', exist_ok=True)
    os.makedirs(f'{out}/states', exist_ok=True)

    params, cfg_name = load_config(family, stage)
    data, ddir = build_data(protocol, family, stage, unit)
    device = 'cpu'
    if need_graphs:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_epochs, patience = EPOCHS[stage]
    if SMOKE:
        max_epochs, patience = 1, 1

    init_state = None
    if stage == 2 and need_graphs:
        sp = f'{out}/states/{family}_s1_u{unit}.pt'
        if not os.path.exists(sp):
            # a silent from-scratch Stage 2 would not be the published procedure
            print(f'missing {sp} -- run Stage 1 for this unit first')
            sys.exit(1)
        import torch
        init_state = torch.load(sp, map_location=device, weights_only=False)
        print(f'  warm start from {sp}')

    print(f'{protocol}/{family} stage {stage} unit {unit}: device={device} '
          f'config={cfg_name} data={ddir}')
    print(f'  train={len(data.idx["train"])} (DILI+ {int(data.labels("train").sum())}) '
          f'val={len(data.idx["val"])} itest={len(data.idx["itest"])} '
          f'Held-223={len(data.idx["test"])}', flush=True)

    t0 = time.time()
    val_auc, probs, model = trainer.run(
        family, params, data, predict_sets=('val', 'itest', 'test'), device=device,
        max_epochs=max_epochs, patience=patience, init_state=init_state)
    print(f'  val AUC={val_auc:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    for s in ('test', 'itest'):
        np.save(f'{out}/probs/{family}_s{stage}_u{unit}_{s}.npy',
                probs[s].astype(np.float32))
    if need_graphs and (stage == 1 or os.environ.get('DILI_SAVE_STAGE2') == '1'):
        # stage-2 weights are normally discarded; the explainability rerun needs
        # them, so they are saved when DILI_SAVE_STAGE2=1
        import torch
        torch.save(model.state_dict(), f'{out}/states/{family}_s{stage}_u{unit}.pt')
    elif not need_graphs and os.environ.get('DILI_SAVE_CLASSICAL') == '1':
        # the fitted RF/SVM/XGB estimators are normally discarded too -- only
        # their probabilities are kept. The consensus SubgraphX run has to
        # re-score masked substructures, which needs the estimator itself.
        import joblib
        joblib.dump(model, f'{out}/states/{family}_s{stage}_u{unit}.joblib')

    rows = []
    for cpt in (0.5, 0.8):
        for split in ('val', 'itest', 'test'):
            rows.append({**evaluate(data.labels(split), probs[split], family, cpt),
                         'Protocol': protocol, 'Stage': stage, 'Unit': unit,
                         'Split': split})
    df = pd.DataFrame(rows)[['Protocol', 'Stage', 'Unit', 'Split'] + METRIC_COLS]
    df.to_csv(f'{out}/{family}_s{stage}_u{unit}_metrics.csv', index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    if (len(sys.argv) != 5 or sys.argv[1] not in PROTOCOLS
            or sys.argv[2] not in ALL_FAMILIES or sys.argv[3] not in ('1', '2')):
        print(f'Usage: python {os.path.basename(__file__)} '
              f'<{"|".join(PROTOCOLS)}> <{"|".join(ALL_FAMILIES)}> <1|2> <unit>')
        sys.exit(1)
    protocol, family, stage, unit = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    if not 0 <= unit < PROTOCOLS[protocol]:
        print(f'unit must be 0..{PROTOCOLS[protocol] - 1} for {protocol}')
        sys.exit(1)
    if not os.path.exists(f'{unit_dir(protocol, unit)}/idx_train.npy'):
        print('splits missing -- run prepare_splits.py first')
        sys.exit(1)
    main(protocol, family, stage, unit)
