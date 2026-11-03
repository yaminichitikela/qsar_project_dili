"""Trains MV-GNN^cross on clin_train, selects the best of 3 configs by
clin_val ROC-AUC, then evaluates the winner on held-out-223 -- same
train/val/test roles as the RF/XGB/Graphormer/GraphGPS models already in
proxy_to_dili_consistent copy 2.ipynb, so results are directly comparable in
the same Table 6.

Requires prepare_dili_data.py to have been run first (loads from ./data/).

Usage: python train_mvgnn.py
"""
import copy
import os

import numpy as np
import pandas as pd
import torch

from eval_utils import evaluate, METRIC_COLS
from mvgnn_model import (MVGNNCross, MVGNN_CONFIGS, MAX_EPOCHS, PATIENCE,
                          make_paired_loader, mvgnn_train_epoch, mvgnn_eval)

SEED = 42
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

atom_train = torch.load(os.path.join(DATA_DIR, 'atom_train.pt'), weights_only=False)
atom_val = torch.load(os.path.join(DATA_DIR, 'atom_val.pt'), weights_only=False)
atom_test = torch.load(os.path.join(DATA_DIR, 'atom_test.pt'), weights_only=False)
bond_train = torch.load(os.path.join(DATA_DIR, 'bond_train.pt'), weights_only=False)
bond_val = torch.load(os.path.join(DATA_DIR, 'bond_val.pt'), weights_only=False)
bond_test = torch.load(os.path.join(DATA_DIR, 'bond_test.pt'), weights_only=False)
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))

ATOM_DIM = atom_train[0].x.shape[1]
ATOM_EDGE_DIM = atom_train[0].edge_attr.shape[1]
BOND_DIM = bond_train[0].x.shape[1]
BOND_EDGE_DIM = bond_train[0].edge_attr.shape[1]
print(f'Atom graph: node_dim={ATOM_DIM} edge_dim={ATOM_EDGE_DIM}')
print(f'Bond graph: node_dim={BOND_DIM} edge_dim={BOND_EDGE_DIM}')

BATCH = 64
train_loader = make_paired_loader(atom_train, bond_train, BATCH, shuffle=True)
val_loader = make_paired_loader(atom_val, bond_val, BATCH, shuffle=False)
test_loader = make_paired_loader(atom_test, bond_test, BATCH, shuffle=False)

pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], device=DEVICE)
print(f'Train: {len(atom_train)}  Val: {len(atom_val)}  Test: {len(atom_test)}')
print(f'pos_weight: {pos_weight.item():.3f}')

best_auc, best_state, best_cfg = 0, None, {}

for cfg in MVGNN_CONFIGS:
    model = MVGNNCross(ATOM_DIM, ATOM_EDGE_DIM, BOND_DIM, BOND_EDGE_DIM,
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

print(f'>> Best MV-GNN^cross: val AUC={best_auc:.4f}  cfg={best_cfg}')

model = MVGNNCross(ATOM_DIM, ATOM_EDGE_DIM, BOND_DIM, BOND_EDGE_DIM,
                    hidden=best_cfg['hidden'], n_layers=best_cfg['n_layers'],
                    dropout=best_cfg['dropout']).to(DEVICE)
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
