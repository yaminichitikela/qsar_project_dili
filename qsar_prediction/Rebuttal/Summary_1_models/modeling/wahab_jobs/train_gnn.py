"""GNN training, one architecture per job so all 4 can run in parallel.
Usage: python train_gnn.py GCN   (or GAT / GIN / SAGE)
Requires prepare_split.py to have been run first (loads from ./data/).
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
from gnn_model import (MolGNN, ATOM_DIM, GNN_CONFIGS, MAX_EPOCHS, PATIENCE,
                        gnn_train_epoch, gnn_eval)

SEED = 42
torch.manual_seed(SEED)

if len(sys.argv) != 2 or sys.argv[1] not in GNN_CONFIGS:
    print(f'Usage: python {sys.argv[0]} <GCN|GAT|GIN|SAGE>')
    sys.exit(1)
ARCH = sys.argv[1]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Architecture: {ARCH}  Device: {DEVICE}')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

graphs_train = torch.load(os.path.join(DATA_DIR, 'graphs_train.pt'), weights_only=False)
graphs_val = torch.load(os.path.join(DATA_DIR, 'graphs_val.pt'), weights_only=False)
graphs_test = torch.load(os.path.join(DATA_DIR, 'graphs_test.pt'), weights_only=False)

BATCH = 64
train_loader = DataLoader(graphs_train, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(graphs_val, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(graphs_test, batch_size=BATCH, shuffle=False)

y_train_np = np.array([g.y.item() for g in graphs_train])
pos_weight = torch.tensor([(y_train_np == 0).sum() / (y_train_np == 1).sum()], device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f'Train graphs: {len(graphs_train)}  Val: {len(graphs_val)}  Test: {len(graphs_test)}')
print(f'pos_weight: {pos_weight.item():.3f} (1.0 = balanced)')

configs = GNN_CONFIGS[ARCH]
best_auc, best_state, best_cfg = 0, None, {}
train_hist, val_hist = [], []

for cfg in configs:
    heads = cfg.pop('heads', 4)
    model = MolGNN(ATOM_DIM, hidden=cfg['hidden'], n_layers=cfg['n_layers'],
                   dropout=cfg['dropout'], conv_type=ARCH, heads=heads).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10)
    cfg['heads'] = heads

    best_ep_auc, no_improv, best_ep_state = 0, 0, None
    tr_aucs, va_aucs = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        gnn_train_epoch(model, train_loader, opt, criterion, DEVICE)
        _, _, tr_auc = gnn_eval(model, train_loader, DEVICE)
        _, _, va_auc = gnn_eval(model, val_loader, DEVICE)
        sched.step(va_auc)
        tr_aucs.append(tr_auc)
        va_aucs.append(va_auc)
        if va_auc > best_ep_auc:
            best_ep_auc = va_auc
            best_ep_state = copy.deepcopy(model.state_dict())
            no_improv = 0
        else:
            no_improv += 1
        if no_improv >= PATIENCE:
            break

    print(f'  {cfg}  val AUC={best_ep_auc:.4f}  stopped@ep{len(tr_aucs)}')
    if best_ep_auc > best_auc:
        best_auc, best_state, best_cfg = best_ep_auc, best_ep_state, cfg.copy()
        train_hist, val_hist = tr_aucs, va_aucs

print(f'>> Best {ARCH}: val AUC={best_auc:.4f}  cfg={best_cfg}')

# ── final eval on val + test with the best config/state ────────────────────
model = MolGNN(ATOM_DIM, hidden=best_cfg['hidden'], n_layers=best_cfg['n_layers'],
               dropout=best_cfg['dropout'], conv_type=ARCH, heads=best_cfg.get('heads', 4)).to(DEVICE)
model.load_state_dict(best_state)

prob_val, y_val, _ = gnn_eval(model, val_loader, DEVICE)
prob_test, y_test, _ = gnn_eval(model, test_loader, DEVICE)

rows = []
for cpt in [0.5, 0.8]:
    rows.append({**evaluate(y_val, prob_val, ARCH, cpt), 'Split': 'Val'})
    rows.append({**evaluate(y_test, prob_test, ARCH, cpt), 'Split': 'Test'})

df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
df.to_csv(os.path.join(OUT_DIR, f'{ARCH.lower()}_results.csv'), index=False)
np.save(os.path.join(OUT_DIR, f'{ARCH.lower()}_test_probs.npy'), prob_test)
torch.save({'state_dict': best_state, 'config': best_cfg},
           os.path.join(OUT_DIR, f'{ARCH.lower()}_model.pt'))
print(df)
print(f'Saved -> {OUT_DIR}/{ARCH.lower()}_results.csv')
