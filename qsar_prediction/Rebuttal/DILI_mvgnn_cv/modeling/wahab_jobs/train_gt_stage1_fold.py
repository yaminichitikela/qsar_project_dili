"""GT family Stage-1 ("Proxy-only") official run for ONE (arch, fold):
retrains the winning config (select_gt_stage1_config.py) on that fold's
clin_train alone, early-stopping on that fold's clin_val, saving the
checkpoint. Same role as train_gnn_stage1_fold.py.

Usage: python train_gt_stage1_fold.py <Graphormer|GraphGPS> <fold_idx 0-4>
Requires prepare_dili_cv_data.py AND select_gt_stage1_config.py <arch> to
have run first.
Outputs (./results/gt_stage1_folds/<arch>/): fold_<i>.pt, fold_<i>.npz
"""
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gt_model import GraphormerModel, GraphGPSModel, PE_DIM, MAX_EPOCHS, PATIENCE, gt_train, gt_eval

SEED = 42
BATCH = 64
ARCHS = ('Graphormer', 'GraphGPS')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def build_model(arch, atom_dim, cfg):
    if arch == 'Graphormer':
        return GraphormerModel(atom_dim, cfg['hidden'], cfg['num_heads'], cfg['num_layers'], cfg['dropout'])
    return GraphGPSModel(atom_dim, cfg['hidden'], PE_DIM, cfg['num_heads'], cfg['num_layers'], cfg['dropout'])


def main(arch, fold):
    torch.manual_seed(SEED)
    has_pe = (arch == 'GraphGPS')

    search_dir = os.path.join(os.path.dirname(__file__), 'results', 'gt_cv_search', arch.lower())
    with open(os.path.join(search_dir, 'winning_config.json')) as f:
        winner = json.load(f)
    cfg = winner['config']
    print(f'{arch} Fold {fold}: winning config (idx {winner["config_idx"]}) = {cfg}')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')

    out_dir = os.path.join(os.path.dirname(__file__), 'results', 'gt_stage1_folds', arch.lower())
    os.makedirs(out_dir, exist_ok=True)

    g_test = torch.load(os.path.join(DATA_DIR, 'gt_test.pt'), weights_only=False)
    test_loader = DataLoader(g_test, batch_size=BATCH, shuffle=False)

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

    best_auc, no_improv, best_state = 0, 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        gt_train(model, opt, train_loader, criterion, DEVICE, has_pe)
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

    print(f'>> {arch} Fold {fold} Stage-1 best val AUC={best_auc:.4f}')

    model.load_state_dict(best_state)
    prob_test, y_test, _ = gt_eval(model, test_loader, DEVICE, has_pe)

    torch.save({'state_dict': best_state, 'config': cfg, 'atom_dim': ATOM_DIM,
                'fold': fold, 'val_auc': best_auc}, os.path.join(out_dir, f'fold_{fold}.pt'))
    np.savez(os.path.join(out_dir, f'fold_{fold}.npz'),
              val_auc=best_auc, test_probs=prob_test, y_test=y_test)
    print(f'Saved -> {out_dir}/fold_{fold}.pt, fold_{fold}.npz')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ARCHS or not sys.argv[2].isdigit() or not (0 <= int(sys.argv[2]) < 5):
        print(f'Usage: python {sys.argv[0]} <{"|".join(ARCHS)}> <fold_idx 0-4>')
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]))
