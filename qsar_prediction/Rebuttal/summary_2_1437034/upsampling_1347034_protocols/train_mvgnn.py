"""MV-GNN^cross on Caspase-3/7 HepG2, AID 1347034 only, up-sampled training set.
Usage: python train_mvgnn.py <loo|scaffold> 
Requires the matching prepare_*_upsample.py and prepare_mvgnn_bonds_protocol.py to have been run first.
Same hyperparameter grids, selection rule and metrics as the published
../modeling/wahab_jobs/train_mvgnn_<protocol> scripts; one run per split unit, then
mean +/- std across units.
"""
import copy
import os

import numpy as np
import pandas as pd
import torch

from eval_utils import evaluate, METRIC_COLS
from protocols import from_argv, unit_dir, prob_name

P = from_argv('')
from mvgnn_model import (MVGNNCross, ATOM_DIM, BOND_DIM, MVGNN_CONFIGS,
                          MAX_EPOCHS, PATIENCE, make_paired_loader,
                          mvgnn_train_epoch, mvgnn_eval)

SEED = 42
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

OUT_DIR = P['out_dir']
os.makedirs(OUT_DIR, exist_ok=True)

BATCH = 64
repeat_rows = []
per_repeat_test_metrics = {0.5: [], 0.8: []}

for rep in range(P['n']):
    rep_dir = unit_dir(P, rep)
    atom_train = torch.load(os.path.join(rep_dir, 'graphs_train.pt'), weights_only=False)
    atom_val = torch.load(os.path.join(rep_dir, 'graphs_val.pt'), weights_only=False)
    atom_test = torch.load(os.path.join(rep_dir, 'graphs_test.pt'), weights_only=False)
    bond_train = torch.load(os.path.join(rep_dir, 'graphs_train_bond.pt'), weights_only=False)
    bond_val = torch.load(os.path.join(rep_dir, 'graphs_val_bond.pt'), weights_only=False)
    bond_test = torch.load(os.path.join(rep_dir, 'graphs_test_bond.pt'), weights_only=False)

    train_loader = make_paired_loader(atom_train, bond_train, BATCH, shuffle=True)
    val_loader = make_paired_loader(atom_val, bond_val, BATCH, shuffle=False)
    test_loader = make_paired_loader(atom_test, bond_test, BATCH, shuffle=False)

    y_train_np = np.array([g.y.item() for g in atom_train])
    pos_weight = torch.tensor([(y_train_np == 0).sum() / (y_train_np == 1).sum()], device=DEVICE)

    best_auc, best_state, best_cfg = 0, None, {}

    for cfg in MVGNN_CONFIGS:
        model = MVGNNCross(ATOM_DIM, BOND_DIM, BOND_DIM, ATOM_DIM,
                            hidden=cfg['hidden'], n_layers=cfg['n_layers'], dropout=cfg['dropout']).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)

        best_ep_auc, no_improv, best_ep_state = 0, 0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            mvgnn_train_epoch(model, train_loader, opt, pos_weight, cfg['lam'], DEVICE)
            _, _, va_auc = mvgnn_eval(model, val_loader, DEVICE)
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

    model = MVGNNCross(ATOM_DIM, BOND_DIM, BOND_DIM, ATOM_DIM,
                        hidden=best_cfg['hidden'], n_layers=best_cfg['n_layers'], dropout=best_cfg['dropout']).to(DEVICE)
    model.load_state_dict(best_state)
    prob_val, y_val, _ = mvgnn_eval(model, val_loader, DEVICE)
    prob_test, y_test, _ = mvgnn_eval(model, test_loader, DEVICE)

    probs_dir = os.path.join(OUT_DIR, 'probs')
    os.makedirs(probs_dir, exist_ok=True)
    np.save(os.path.join(probs_dir, prob_name(P, 'mvgnn', rep)), prob_test)

    for cpt in [0.5, 0.8]:
        row_val = {**evaluate(y_val, prob_val, 'MV-GNN-cross', cpt), 'Split': 'Val', P['unit_col']: rep}
        row_test = {**evaluate(y_test, prob_test, 'MV-GNN-cross', cpt), 'Split': 'Test', P['unit_col']: rep}
        repeat_rows.append(row_val)
        repeat_rows.append(row_test)
        per_repeat_test_metrics[cpt].append(row_test)

df_reps = pd.DataFrame(repeat_rows)[[P['unit_col'], 'Split'] + METRIC_COLS]
df_reps.to_csv(os.path.join(OUT_DIR, 'mvgnn_per_unit.csv'), index=False)

summary_rows = []
for cpt in [0.5, 0.8]:
    test_df = pd.DataFrame(per_repeat_test_metrics[cpt])
    numeric_cols = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
    mean_row = test_df[numeric_cols].mean().round(4)
    std_row = test_df[numeric_cols].std().round(4)
    summary_rows.append({'Model': 'MV-GNN-cross', 'CPT': cpt, 'Stat': 'mean', **mean_row.to_dict()})
    summary_rows.append({'Model': 'MV-GNN-cross', 'CPT': cpt, 'Stat': 'std', **std_row.to_dict()})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, 'mvgnn_summary.csv'), index=False)
print(f'\n== Mean +/- std across {P["n"]} {P["unit"]}s (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/mvgnn_per_unit.csv and mvgnn_summary.csv')
