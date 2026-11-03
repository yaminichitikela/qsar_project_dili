"""Graph Transformer training, one architecture per job so both can run in
parallel -- same pattern as train_gnn.py.
Usage: python train_gt.py Graphormer   (or GraphGPS)
Requires prepare_split.py (both archs) and, for GraphGPS, prepare_gt_pe.py to
have been run first.
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
torch.manual_seed(SEED)

if len(sys.argv) != 2 or sys.argv[1] not in GT_CONFIGS:
    print(f'Usage: python {sys.argv[0]} <Graphormer|GraphGPS>')
    sys.exit(1)
ARCH = sys.argv[1]
HAS_PE = (ARCH == 'GraphGPS')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Architecture: {ARCH}  Device: {DEVICE}')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

suffix = '_gps' if HAS_PE else ''
graphs_train = torch.load(os.path.join(DATA_DIR, f'graphs_train{suffix}.pt'), weights_only=False)
graphs_val = torch.load(os.path.join(DATA_DIR, f'graphs_val{suffix}.pt'), weights_only=False)
graphs_test = torch.load(os.path.join(DATA_DIR, f'graphs_test{suffix}.pt'), weights_only=False)

BATCH = 64
train_loader = DataLoader(graphs_train, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(graphs_val, batch_size=BATCH, shuffle=False)
test_loader = DataLoader(graphs_test, batch_size=BATCH, shuffle=False)

y_train_np = np.array([g.y.item() for g in graphs_train])
pos_weight = torch.tensor([(y_train_np == 0).sum() / (y_train_np == 1).sum()], device=DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f'Train graphs: {len(graphs_train)}  Val: {len(graphs_val)}  Test: {len(graphs_test)}')
print(f'pos_weight: {pos_weight.item():.3f} (1.0 = balanced)')

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

    print(f'  {cfg}  val AUC={best_ep_auc:.4f}  stopped@ep{epoch}')
    if best_ep_auc > best_auc:
        best_auc, best_state, best_cfg = best_ep_auc, best_ep_state, cfg.copy()

print(f'>> Best {ARCH}: val AUC={best_auc:.4f}  cfg={best_cfg}')

# ── final eval on val + test with the best config/state ────────────────────
if ARCH == 'Graphormer':
    model = GraphormerModel(ATOM_DIM, best_cfg['hidden'], best_cfg['num_heads'],
                             best_cfg['num_layers'], best_cfg['dropout']).to(DEVICE)
else:
    model = GraphGPSModel(ATOM_DIM, best_cfg['hidden'], PE_DIM, best_cfg['num_heads'],
                           best_cfg['num_layers'], best_cfg['dropout']).to(DEVICE)
model.load_state_dict(best_state)

prob_val, y_val, _ = gt_eval(model, val_loader, DEVICE, HAS_PE)
prob_test, y_test, _ = gt_eval(model, test_loader, DEVICE, HAS_PE)

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
