"""Graph Transformer training (Graphormer/GraphGPS), one architecture per
job. Self-contained in this folder.

Usage: python train_gt_sampling.py <Graphormer|GraphGPS> <hepg2|cho_k1> <v1_ratio_1to5|v2_similarity|v3_diversity>
(not v4_smote -- SMOTE has no graphs, classical models only)
For GraphGPS, prepare_gt_pe_sampling.py <dataset> <variant> must be run first.
"""
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from common import CASPASE_DATASETS, DATA_ROOT
from eval_utils import evaluate, METRIC_COLS
from gt_model import (GraphormerModel, GraphGPSModel, ATOM_DIM, PE_DIM,
                       GT_CONFIGS, MAX_EPOCHS, PATIENCE, gt_train, gt_eval)

VALID_VARIANTS = ['v1_ratio_1to5', 'v2_similarity', 'v3_diversity']

if (len(sys.argv) != 4 or sys.argv[1] not in GT_CONFIGS
        or sys.argv[2] not in CASPASE_DATASETS or sys.argv[3] not in VALID_VARIANTS):
    print(f'Usage: python {sys.argv[0]} <Graphormer|GraphGPS> <{"|".join(CASPASE_DATASETS)}> <{"|".join(VALID_VARIANTS)}>')
    sys.exit(1)
ARCH, DATASET_NAME, VARIANT = sys.argv[1], sys.argv[2], sys.argv[3]
HAS_PE = (ARCH == 'GraphGPS')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[{DATASET_NAME}/{VARIANT}] Architecture: {ARCH}  Device: {DEVICE}')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(DATA_ROOT, DATASET_NAME, VARIANT)
OUT_DIR = os.path.join(HERE, 'results_sampling', DATASET_NAME, VARIANT)
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
print(df)
print(f'Saved -> {OUT_DIR}/{ARCH.lower()}_results.csv')
