"""GNN (GCN / GAT / GIN / SAGE) on Caspase-3/7 CHO-K1 (AID 1347037), 1:5 downsampled training set.
Usage: python train_gnn.py <scaffold> <GCN|GAT|GIN|SAGE>
Requires prepare_scaffold_1to5.py to have been run first.
Same hyperparameter grids, selection rule and metrics as the published
../modeling/wahab_jobs/train_gnn_<protocol> scripts; one run per split unit, then
mean +/- std across units.
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
from protocols import from_argv, unit_dir, prob_name

P = from_argv('<GCN|GAT|GIN|SAGE>')
from gnn_model import (MolGNN, ATOM_DIM, GNN_CONFIGS, MAX_EPOCHS, PATIENCE,
                        gnn_train_epoch, gnn_eval)

SEED = 42
torch.manual_seed(SEED)

if len(sys.argv) != 2 or sys.argv[1] not in GNN_CONFIGS:
    print(f'Usage: python {sys.argv[0]} <scaffold> <GCN|GAT|GIN|SAGE>')
    sys.exit(1)
ARCH = sys.argv[1]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Architecture: {ARCH}  Device: {DEVICE}')

OUT_DIR = P['out_dir']
os.makedirs(OUT_DIR, exist_ok=True)

BATCH = 64
repeat_rows = []
per_repeat_test_metrics = {0.5: [], 0.8: []}

for rep in range(P['n']):
    rep_dir = unit_dir(P, rep)
    graphs_train = torch.load(os.path.join(rep_dir, 'graphs_train.pt'), weights_only=False)
    graphs_val = torch.load(os.path.join(rep_dir, 'graphs_val.pt'), weights_only=False)
    graphs_test = torch.load(os.path.join(rep_dir, 'graphs_test.pt'), weights_only=False)

    train_loader = DataLoader(graphs_train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(graphs_val, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(graphs_test, batch_size=BATCH, shuffle=False)

    y_train_np = np.array([g.y.item() for g in graphs_train])
    pos_weight = torch.tensor([(y_train_np == 0).sum() / (y_train_np == 1).sum()], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    configs = [cfg.copy() for cfg in GNN_CONFIGS[ARCH]]
    best_auc, best_state, best_cfg = 0, None, {}

    for cfg in configs:
        heads = cfg.pop('heads', 4)
        model = MolGNN(ATOM_DIM, hidden=cfg['hidden'], n_layers=cfg['n_layers'],
                       dropout=cfg['dropout'], conv_type=ARCH, heads=heads).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)
        cfg['heads'] = heads

        best_ep_auc, no_improv, best_ep_state = 0, 0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            gnn_train_epoch(model, train_loader, opt, criterion, DEVICE)
            _, _, va_auc = gnn_eval(model, val_loader, DEVICE)
            sched.step(va_auc)
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

    print(f'{P["unit_col"]} {rep}: best val AUC={best_auc:.4f}  cfg={best_cfg}')

    model = MolGNN(ATOM_DIM, hidden=best_cfg['hidden'], n_layers=best_cfg['n_layers'],
                   dropout=best_cfg['dropout'], conv_type=ARCH, heads=best_cfg.get('heads', 4)).to(DEVICE)
    model.load_state_dict(best_state)
    prob_val, y_val, _ = gnn_eval(model, val_loader, DEVICE)
    prob_test, y_test, _ = gnn_eval(model, test_loader, DEVICE)

    # save this unit's winning test-set probs, needed by compute_consensus.py
    probs_dir = os.path.join(OUT_DIR, 'probs')
    os.makedirs(probs_dir, exist_ok=True)
    np.save(os.path.join(probs_dir, prob_name(P, ARCH.lower(), rep)), prob_test)

    for cpt in [0.5, 0.8]:
        row_val = {**evaluate(y_val, prob_val, ARCH, cpt), 'Split': 'Val', P['unit_col']: rep}
        row_test = {**evaluate(y_test, prob_test, ARCH, cpt), 'Split': 'Test', P['unit_col']: rep}
        repeat_rows.append(row_val)
        repeat_rows.append(row_test)
        per_repeat_test_metrics[cpt].append(row_test)

df_reps = pd.DataFrame(repeat_rows)[[P['unit_col'], 'Split'] + METRIC_COLS]
df_reps.to_csv(os.path.join(OUT_DIR, f'{ARCH.lower()}_per_unit.csv'), index=False)

summary_rows = []
for cpt in [0.5, 0.8]:
    test_df = pd.DataFrame(per_repeat_test_metrics[cpt])
    numeric_cols = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
    mean_row = test_df[numeric_cols].mean().round(4)
    std_row = test_df[numeric_cols].std().round(4)
    summary_rows.append({'Model': ARCH, 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
    summary_rows.append({'Model': ARCH, 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, f'{ARCH.lower()}_summary.csv'), index=False)
print(f'\n== Mean +/- std across {P["n"]} {P["unit"]}s (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/{ARCH.lower()}_per_unit.csv and {ARCH.lower()}_summary.csv')
