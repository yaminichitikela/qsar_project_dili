"""Graph Transformer family (Graphormer/GraphGPS) Stage-1 SEARCH phase: runs
5-fold CV for ONE candidate hyperparameter config of ONE architecture
(GT_CONFIGS[arch][idx]). Same pattern as train_gnn_cv_config.py, using
gt_model.py's GraphormerModel/GraphGPSModel + GT_CONFIGS grid.

Usage: python train_gt_cv_config.py <Graphormer|GraphGPS> <config_idx>
Requires prepare_dili_cv_data.py to have been run first (loads from ./data/).
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gt_model import GraphormerModel, GraphGPSModel, GT_CONFIGS, PE_DIM, MAX_EPOCHS, PATIENCE, gt_train, gt_eval

SEED = 42
N_FOLDS = 5
BATCH = 64
ARCHS = ('Graphormer', 'GraphGPS')


def build_model(arch, atom_dim, cfg):
    if arch == 'Graphormer':
        return GraphormerModel(atom_dim, cfg['hidden'], cfg['num_heads'], cfg['num_layers'], cfg['dropout'])
    return GraphGPSModel(atom_dim, cfg['hidden'], PE_DIM, cfg['num_heads'], cfg['num_layers'], cfg['dropout'])


def main(arch, cfg_idx):
    torch.manual_seed(SEED)
    cfg = GT_CONFIGS[arch][cfg_idx]
    has_pe = (arch == 'GraphGPS')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{arch} Config {cfg_idx}: {cfg}  Device: {DEVICE}')

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'gt_cv_search', arch.lower())
    os.makedirs(OUT_DIR, exist_ok=True)

    g_test = torch.load(os.path.join(DATA_DIR, 'gt_test.pt'), weights_only=False)
    test_loader = DataLoader(g_test, batch_size=BATCH, shuffle=False)

    fold_val_aucs = []
    fold_test_probs = []

    for fold in range(N_FOLDS):
        fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
        g_tr = torch.load(os.path.join(fold_dir, 'gt_train.pt'), weights_only=False)
        g_va = torch.load(os.path.join(fold_dir, 'gt_val.pt'), weights_only=False)
        ATOM_DIM = g_tr[0].x.shape[1]

        train_loader = DataLoader(g_tr, batch_size=BATCH, shuffle=True)
        val_loader = DataLoader(g_va, batch_size=BATCH, shuffle=False)
        y_tr_np = np.array([g.y.item() for g in g_tr])
        pos_weight = torch.tensor([(y_tr_np == 0).sum() / (y_tr_np == 1).sum()], device=DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model = build_model(arch, ATOM_DIM, cfg).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

        best_ep_auc, no_improv, best_ep_state = 0, 0, None
        for epoch in range(1, MAX_EPOCHS + 1):
            gt_train(model, opt, train_loader, criterion, DEVICE, has_pe)
            _, _, va_auc = gt_eval(model, val_loader, DEVICE, has_pe)
            if va_auc > best_ep_auc:
                best_ep_auc, best_ep_state, no_improv = va_auc, copy.deepcopy(model.state_dict()), 0
            else:
                no_improv += 1
            if no_improv >= PATIENCE:
                break

        print(f'  fold {fold}: val AUC={best_ep_auc:.4f}  stopped@ep{epoch}')
        fold_val_aucs.append(best_ep_auc)

        model.load_state_dict(best_ep_state)
        prob_test, _, _ = gt_eval(model, test_loader, DEVICE, has_pe)
        fold_test_probs.append(prob_test)

    mean_cv_auc = float(np.mean(fold_val_aucs))
    std_cv_auc = float(np.std(fold_val_aucs))
    print(f'>> {arch} Config {cfg_idx} CV val AUC = {mean_cv_auc:.4f} +/- {std_cv_auc:.4f}')

    np.savez(os.path.join(OUT_DIR, f'config_{cfg_idx}.npz'),
             cfg=cfg, fold_val_aucs=fold_val_aucs, fold_test_probs=np.array(fold_test_probs),
             mean_cv_auc=mean_cv_auc, std_cv_auc=std_cv_auc)
    print(f'Saved -> {OUT_DIR}/config_{cfg_idx}.npz')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ARCHS or not sys.argv[2].isdigit():
        print(f'Usage: python {sys.argv[0]} <{"|".join(ARCHS)}> <config_idx>')
        sys.exit(1)
    arch = sys.argv[1]
    cfg_idx = int(sys.argv[2])
    if not (0 <= cfg_idx < len(GT_CONFIGS[arch])):
        print(f'config_idx must be in [0, {len(GT_CONFIGS[arch]) - 1}] for {arch}')
        sys.exit(1)
    main(arch, cfg_idx)
