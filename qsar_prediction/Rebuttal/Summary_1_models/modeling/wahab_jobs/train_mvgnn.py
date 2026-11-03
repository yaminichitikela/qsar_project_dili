"""MV-GNN^cross training, Random split (8:1:1).
Requires prepare_split.py and prepare_mvgnn_bonds.py to have been run first.

Usage: python train_mvgnn.py
"""
import copy
import os

import numpy as np
import pandas as pd
import torch

from eval_utils import evaluate, METRIC_COLS
from mvgnn_model import (MVGNNCross, ATOM_DIM, BOND_DIM, MVGNN_CONFIGS,
                          MAX_EPOCHS, PATIENCE, make_paired_loader,
                          mvgnn_train_epoch, mvgnn_eval)

SEED = 42
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

atom_train = torch.load(os.path.join(DATA_DIR, 'graphs_train.pt'), weights_only=False)
atom_val = torch.load(os.path.join(DATA_DIR, 'graphs_val.pt'), weights_only=False)
atom_test = torch.load(os.path.join(DATA_DIR, 'graphs_test.pt'), weights_only=False)
bond_train = torch.load(os.path.join(DATA_DIR, 'graphs_train_bond.pt'), weights_only=False)
bond_val = torch.load(os.path.join(DATA_DIR, 'graphs_val_bond.pt'), weights_only=False)
bond_test = torch.load(os.path.join(DATA_DIR, 'graphs_test_bond.pt'), weights_only=False)

BATCH = 64
train_loader = make_paired_loader(atom_train, bond_train, BATCH, shuffle=True)
val_loader = make_paired_loader(atom_val, bond_val, BATCH, shuffle=False)
test_loader = make_paired_loader(atom_test, bond_test, BATCH, shuffle=False)

y_train_np = np.array([g.y.item() for g in atom_train])
pos_weight = torch.tensor([(y_train_np == 0).sum() / (y_train_np == 1).sum()], device=DEVICE)
print(f'Train graphs: {len(atom_train)}  Val: {len(atom_val)}  Test: {len(atom_test)}')
print(f'pos_weight: {pos_weight.item():.3f} (1.0 = balanced)')

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

    print(f'  {cfg}  val AUC={best_ep_auc:.4f}  stopped@ep{epoch}')
    if best_ep_auc > best_auc:
        best_auc, best_state, best_cfg = best_ep_auc, best_ep_state, cfg.copy()

print(f'>> Best MV-GNN-cross: val AUC={best_auc:.4f}  cfg={best_cfg}')

model = MVGNNCross(ATOM_DIM, BOND_DIM, BOND_DIM, ATOM_DIM,
                    hidden=best_cfg['hidden'], n_layers=best_cfg['n_layers'], dropout=best_cfg['dropout']).to(DEVICE)
model.load_state_dict(best_state)

prob_val, y_val, _ = mvgnn_eval(model, val_loader, DEVICE)
prob_test, y_test, _ = mvgnn_eval(model, test_loader, DEVICE)

rows = []
for cpt in [0.5, 0.8]:
    rows.append({**evaluate(y_val, prob_val, 'MV-GNN-cross', cpt), 'Split': 'Val'})
    rows.append({**evaluate(y_test, prob_test, 'MV-GNN-cross', cpt), 'Split': 'Test'})

df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
df.to_csv(os.path.join(OUT_DIR, 'mvgnn_results.csv'), index=False)
np.save(os.path.join(OUT_DIR, 'mvgnn_test_probs.npy'), prob_test)
torch.save({'state_dict': best_state, 'config': best_cfg},
           os.path.join(OUT_DIR, 'mvgnn_model.pt'))
print(df)
print(f'Saved -> {OUT_DIR}/mvgnn_results.csv')
