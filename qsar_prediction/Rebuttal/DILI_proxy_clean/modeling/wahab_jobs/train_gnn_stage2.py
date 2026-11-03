"""GNN family (GCN/GAT/GIN/SAGE) Stage-2 ("Proxy+DILI"): loads the Stage-1
checkpoint, fine-tunes on clin_train + NCV-888 combined, early-stops on
clin_val, same schedule as Graphormer/GraphGPS/MV-GNN Stage-2
(max_epochs=60, patience=12, lr=5e-4) -- no re-search, same architecture/config
as Stage-1, just more training data, matching how every other graph model in
this table treats Stage-2.

Usage: python train_gnn_stage2.py <GCN|GAT|GIN|SAGE>
Requires prepare_proxy_data_clean.py AND train_gnn_stage1.py <arch> to have run first.
Outputs (./results/): {arch}_stage2_results.csv, {arch}_stage2_test_probs.npy
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
from gnn_model import MolGNN, GNN_CONFIGS, gnn_train_epoch, gnn_eval

SEED = 42
MAX_EPOCHS = 60
PATIENCE = 12
LR = 5e-4
BATCH = 64

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def main(arch):
    torch.manual_seed(SEED)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join(OUT_DIR, f'{arch.lower()}_stage1.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Missing {ckpt_path} -- run train_gnn_stage1.py {arch} first.')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    print(f'Architecture: {arch}  Device: {DEVICE}  Loaded Stage-1 config={cfg}')

    g_train = torch.load(os.path.join(DATA_DIR, 'gt_train.pt'), weights_only=False)
    g_ncv = torch.load(os.path.join(DATA_DIR, 'gt_ncv.pt'), weights_only=False)
    g_val = torch.load(os.path.join(DATA_DIR, 'gt_val.pt'), weights_only=False)
    g_test = torch.load(os.path.join(DATA_DIR, 'gt_test.pt'), weights_only=False)
    g_aug = g_train + g_ncv

    aug_loader = DataLoader(g_aug, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(g_val, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(g_test, batch_size=BATCH, shuffle=False)

    y_aug_np = np.array([g.y.item() for g in g_aug])
    pos_weight = torch.tensor([(y_aug_np == 0).sum() / (y_aug_np == 1).sum()], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f'Stage-2 train (clin_train + NCV-888): {len(g_aug)}  pos_weight={pos_weight.item():.3f}')

    model = MolGNN(ckpt['atom_dim'], hidden=cfg['hidden'], n_layers=cfg['n_layers'],
                    dropout=cfg['dropout'], conv_type=arch, heads=cfg.get('heads', 4)).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg.get('weight_decay', 1e-5))

    best_auc, best_state, no_improv = 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        gnn_train_epoch(model, aug_loader, opt, criterion, DEVICE)
        _, _, va_auc = gnn_eval(model, val_loader, DEVICE)
        if va_auc > best_auc:
            best_auc, best_state, no_improv = va_auc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improv += 1
        if no_improv >= PATIENCE:
            break
    print(f'>> Stage-2 best {arch}: val AUC={best_auc:.4f}  stopped@ep{epoch}')

    model.load_state_dict(best_state)
    prob_val, y_val, _ = gnn_eval(model, val_loader, DEVICE)
    prob_test, y_test, _ = gnn_eval(model, test_loader, DEVICE)

    rows = []
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_val, prob_val, f'{arch} Stage-2 (Proxy+DILI)', cpt), 'Split': 'Val'})
        rows.append({**evaluate(y_test, prob_test, f'{arch} Stage-2 (Proxy+DILI)', cpt), 'Split': 'Test'})

    df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
    df.to_csv(os.path.join(OUT_DIR, f'{arch.lower()}_stage2_results.csv'), index=False)
    np.save(os.path.join(OUT_DIR, f'{arch.lower()}_stage2_test_probs.npy'), prob_test)
    print(df)
    print(f'Saved -> {OUT_DIR}/{arch.lower()}_stage2_results.csv')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in GNN_CONFIGS:
        print(f'Usage: python {sys.argv[0]} <GCN|GAT|GIN|SAGE>')
        sys.exit(1)
    main(sys.argv[1])
