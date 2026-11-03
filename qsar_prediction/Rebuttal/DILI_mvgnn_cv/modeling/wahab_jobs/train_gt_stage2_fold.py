"""GT family Stage-2 ("Proxy+DILI") for ONE (arch, fold): loads that
architecture+fold's Stage-1 checkpoint and fine-tunes on clin_train_fold +
NCV-888 combined, early-stopping on that same fold's clin_val -- same
schedule as DILI_proxy_clean's train_gt_stage2.py (LR=5e-4, max 60 epochs,
patience=12, no re-search).

Usage: python train_gt_stage2_fold.py <Graphormer|GraphGPS> <fold_idx 0-4>
Requires prepare_dili_cv_data.py AND train_gt_stage1_fold.py <arch> <fold>
to have run first.
Outputs (./results/gt_stage2_folds/<arch>/): fold_<i>.pt, fold_<i>.npz
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gt_model import GraphormerModel, GraphGPSModel, PE_DIM, gt_train, gt_eval

SEED = 42
BATCH = 64
MAX_EPOCHS = 60
PATIENCE = 12
LR = 5e-4
ARCHS = ('Graphormer', 'GraphGPS')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def build_model(arch, atom_dim, cfg):
    if arch == 'Graphormer':
        return GraphormerModel(atom_dim, cfg['hidden'], cfg['num_heads'], cfg['num_layers'], cfg['dropout'])
    return GraphGPSModel(atom_dim, cfg['hidden'], PE_DIM, cfg['num_heads'], cfg['num_layers'], cfg['dropout'])


def main(arch, fold):
    torch.manual_seed(SEED)
    has_pe = (arch == 'GraphGPS')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stage1_dir = os.path.join(os.path.dirname(__file__), 'results', 'gt_stage1_folds', arch.lower())
    ckpt_path = os.path.join(stage1_dir, f'fold_{fold}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Missing {ckpt_path} -- run train_gt_stage1_fold.py {arch} {fold} first.')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    print(f'{arch} Fold {fold}: loaded Stage-1 checkpoint, config={cfg}')

    out_dir = os.path.join(os.path.dirname(__file__), 'results', 'gt_stage2_folds', arch.lower())
    os.makedirs(out_dir, exist_ok=True)

    fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
    g_tr = torch.load(os.path.join(fold_dir, 'gt_train.pt'), weights_only=False)
    g_va = torch.load(os.path.join(fold_dir, 'gt_val.pt'), weights_only=False)
    g_ncv = torch.load(os.path.join(DATA_DIR, 'gt_ncv.pt'), weights_only=False)
    g_test = torch.load(os.path.join(DATA_DIR, 'gt_test.pt'), weights_only=False)

    g_aug = g_tr + g_ncv
    print(f'{arch} Fold {fold} Stage-2 train (clin_train + NCV-888): {len(g_aug):,}')

    aug_loader = DataLoader(g_aug, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(g_va, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(g_test, batch_size=BATCH, shuffle=False)

    y_aug_np = np.array([g.y.item() for g in g_aug])
    pos_weight = torch.tensor([(y_aug_np == 0).sum() / (y_aug_np == 1).sum()], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f'pos_weight (Stage-2): {pos_weight.item():.3f}')

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
        if epoch % 10 == 0 or no_improv >= PATIENCE:
            print(f'  ep {epoch:3d}  val AUC={va_auc:.4f}  best={best_auc:.4f}')
        if no_improv >= PATIENCE:
            print(f'  early stop @ ep {epoch}')
            break

    print(f'>> {arch} Fold {fold} Stage-2 best val AUC={best_auc:.4f}')

    model.load_state_dict(best_state)
    prob_test, y_test, _ = gt_eval(model, test_loader, DEVICE, has_pe)

    torch.save({'state_dict': best_state, 'config': cfg, 'atom_dim': ckpt['atom_dim'],
                'fold': fold, 'val_auc': best_auc}, os.path.join(out_dir, f'fold_{fold}.pt'))
    np.savez(os.path.join(out_dir, f'fold_{fold}.npz'),
              val_auc=best_auc, test_probs=prob_test, y_test=y_test)
    print(f'Saved -> {out_dir}/fold_{fold}.pt, fold_{fold}.npz')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ARCHS or not sys.argv[2].isdigit() or not (0 <= int(sys.argv[2]) < 5):
        print(f'Usage: python {sys.argv[0]} <{"|".join(ARCHS)}> <fold_idx 0-4>')
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]))
