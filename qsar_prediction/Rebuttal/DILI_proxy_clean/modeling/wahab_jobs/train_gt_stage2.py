"""Graph Transformers (Graphormer/GraphGPS) Stage-2 ("Proxy+DILI"): loads the
Stage-1 checkpoint, fine-tunes on clin_train + NCV-888 combined, early-stops on
clin_val, standard 0.5/0.8 thresholds -- same schedule as GNN/MV-GNN Stage-2
(max_epochs=60, patience=12, lr=5e-4). No HP re-search, no threshold tuning --
this replaces copy 4.ipynb's GraphGPS-only "OptF1" val-optimized-threshold
retrain, which gave GraphGPS different treatment than every other model in the
table.

Usage: python train_gt_stage2.py <Graphormer|GraphGPS>
Requires prepare_proxy_data_clean.py AND train_gt_stage1.py <arch> to have run first.
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
from gt_model import GraphormerModel, GraphGPSModel, PE_DIM, GT_CONFIGS, gt_train, gt_eval

SEED = 42
MAX_EPOCHS = 60
PATIENCE = 12
LR = 5e-4
BATCH = 64

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def build_model(arch, atom_dim, cfg):
    if arch == 'Graphormer':
        return GraphormerModel(atom_dim, cfg['hidden'], cfg['num_heads'], cfg['num_layers'], cfg['dropout'])
    return GraphGPSModel(atom_dim, cfg['hidden'], PE_DIM, cfg['num_heads'], cfg['num_layers'], cfg['dropout'])


def main(arch):
    torch.manual_seed(SEED)
    has_pe = (arch == 'GraphGPS')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join(OUT_DIR, f'{arch.lower()}_stage1.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Missing {ckpt_path} -- run train_gt_stage1.py {arch} first.')
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

    model = build_model(arch, ckpt['atom_dim'], cfg).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc, best_state, no_improv = 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        gt_train(model, opt, aug_loader, criterion, DEVICE, has_pe)
        _, _, va_auc = gt_eval(model, val_loader, DEVICE, has_pe)
        if va_auc > best_auc:
            best_auc, best_state, no_improv = va_auc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improv += 1
        if no_improv >= PATIENCE:
            break
    print(f'>> Stage-2 best {arch}: val AUC={best_auc:.4f}  stopped@ep{epoch}')

    model.load_state_dict(best_state)
    prob_val, y_val, _ = gt_eval(model, val_loader, DEVICE, has_pe)
    prob_test, y_test, _ = gt_eval(model, test_loader, DEVICE, has_pe)

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
    if len(sys.argv) != 2 or sys.argv[1] not in GT_CONFIGS:
        print(f'Usage: python {sys.argv[0]} <Graphormer|GraphGPS>')
        sys.exit(1)
    main(sys.argv[1])
