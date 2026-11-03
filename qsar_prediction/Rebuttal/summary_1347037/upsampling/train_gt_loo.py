"""Graph Transformer on the Leave-one-out 10-fold split, one architecture per job.
Usage: python train_gt_loo.py <Graphormer|GraphGPS>
Requires prepare_loo_folds.py and prepare_gt_pe_loo.py to have been run first
(loads from ./data_loo/).
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from eval_utils import evaluate, METRIC_COLS
from gt_model import (GraphormerModel, GraphGPSModel, ATOM_DIM, PE_DIM,
                       GT_CONFIGS, MAX_EPOCHS, PATIENCE, gt_train, gt_eval)

SEED = 42
N_FOLDS = 10
torch.manual_seed(SEED)

if len(sys.argv) != 2 or sys.argv[1] not in GT_CONFIGS:
    print(f'Usage: python {sys.argv[0]} <Graphormer|GraphGPS>')
    sys.exit(1)
ARCH = sys.argv[1]
HAS_PE = (ARCH == 'GraphGPS')
suffix = '_gps' if HAS_PE else ''

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Architecture: {ARCH}  Device: {DEVICE}')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_loo')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results_loo')
os.makedirs(OUT_DIR, exist_ok=True)

BATCH = 64
fold_rows = []
per_fold_test_metrics = {0.5: [], 0.8: []}

for fold in range(N_FOLDS):
    fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
    graphs_train = torch.load(os.path.join(fold_dir, f'graphs_train{suffix}.pt'), weights_only=False)
    graphs_val = torch.load(os.path.join(fold_dir, f'graphs_val{suffix}.pt'), weights_only=False)
    graphs_test = torch.load(os.path.join(fold_dir, f'graphs_test{suffix}.pt'), weights_only=False)

    train_loader = DataLoader(graphs_train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(graphs_val, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(graphs_test, batch_size=BATCH, shuffle=False)

    y_train_np = np.array([g.y.item() for g in graphs_train])
    pos_weight = torch.tensor([(y_train_np == 0).sum() / (y_train_np == 1).sum()], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    configs = [cfg.copy() for cfg in GT_CONFIGS[ARCH]]
    best_auc, best_state, best_cfg = 0, None, {}

    for cfg in configs:
        lr = cfg['lr']
        if ARCH == 'Graphormer':
            model = GraphormerModel(ATOM_DIM, cfg['hidden'], cfg['num_heads'],
                                     cfg['num_layers'], cfg['dropout']).to(DEVICE)
        else:
            model = GraphGPSModel(ATOM_DIM, cfg['hidden'], PE_DIM, cfg['num_heads'],
                                   cfg['num_layers'], cfg['dropout']).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        best_ep_auc, no_improv, best_ep_state = 0, 0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            gt_train(model, opt, train_loader, criterion, DEVICE, HAS_PE)
            _, _, va_auc = gt_eval(model, val_loader, DEVICE, HAS_PE)
            if va_auc > best_ep_auc:
                best_ep_auc = va_auc
                best_ep_state = copy.deepcopy(model.state_dict())
                no_improv = 0
            else:
                no_improv += 1
            if no_improv >= PATIENCE:
                break

        if best_ep_auc > best_auc:
            best_auc, best_state, best_cfg = best_ep_auc, best_ep_state, cfg.copy()

    print(f'Fold {fold}: best val AUC={best_auc:.4f}  cfg={best_cfg}')

    if ARCH == 'Graphormer':
        model = GraphormerModel(ATOM_DIM, best_cfg['hidden'], best_cfg['num_heads'],
                                 best_cfg['num_layers'], best_cfg['dropout']).to(DEVICE)
    else:
        model = GraphGPSModel(ATOM_DIM, best_cfg['hidden'], PE_DIM, best_cfg['num_heads'],
                               best_cfg['num_layers'], best_cfg['dropout']).to(DEVICE)
    model.load_state_dict(best_state)
    prob_val, y_val, _ = gt_eval(model, val_loader, DEVICE, HAS_PE)
    prob_test, y_test, _ = gt_eval(model, test_loader, DEVICE, HAS_PE)

    # save this fold's winning test-set probs, needed by compute_consensus_loo.py
    probs_dir = os.path.join(OUT_DIR, 'probs')
    os.makedirs(probs_dir, exist_ok=True)
    np.save(os.path.join(probs_dir, f'{ARCH.lower()}_fold{fold}_test_probs.npy'), prob_test)

    for cpt in [0.5, 0.8]:
        row_val = {**evaluate(y_val, prob_val, ARCH, cpt), 'Split': 'Val', 'Fold': fold}
        row_test = {**evaluate(y_test, prob_test, ARCH, cpt), 'Split': 'Test', 'Fold': fold}
        fold_rows.append(row_val)
        fold_rows.append(row_test)
        per_fold_test_metrics[cpt].append(row_test)

df_folds = pd.DataFrame(fold_rows)[['Fold', 'Split'] + METRIC_COLS]
df_folds.to_csv(os.path.join(OUT_DIR, f'{ARCH.lower()}_loo_per_fold.csv'), index=False)

summary_rows = []
for cpt in [0.5, 0.8]:
    test_df = pd.DataFrame(per_fold_test_metrics[cpt])
    numeric_cols = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
    mean_row = test_df[numeric_cols].mean().round(4)
    std_row = test_df[numeric_cols].std().round(4)
    summary_rows.append({'Model': ARCH, 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
    summary_rows.append({'Model': ARCH, 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, f'{ARCH.lower()}_loo_summary.csv'), index=False)
print('\n== Mean +/- std across 10 folds (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/{ARCH.lower()}_loo_per_fold.csv and {ARCH.lower()}_loo_summary.csv')
