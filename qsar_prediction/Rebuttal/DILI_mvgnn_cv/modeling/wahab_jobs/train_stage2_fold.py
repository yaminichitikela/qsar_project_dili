"""Stage-2 ("Proxy+DILI") for ONE fold: loads that fold's Stage-1 checkpoint
(train_stage1_fold.py) and fine-tunes on clin_train_fold + NCV-888 combined,
early-stopping on that SAME fold's clin_val (NCV-888 only augments training,
never validation) -- same fine-tuning recipe (LR=5e-4, max 60 epochs,
patience=12, all parameters fine-tuned together, no gradual unfreezing) as
DILI_proxy_clean's train_mvgnn_stage2.py.

Requires prepare_dili_cv_data.py AND train_stage1_fold.py <fold> to have run
first.

Usage: python train_stage2_fold.py <fold_idx 0-4>
Outputs (./results/stage2_folds/):
    fold_<i>.pt  -- checkpoint
    fold_<i>.npz -- val_auc, test_probs
"""
import copy
import os
import sys

import numpy as np
import torch

from mvgnn_model import MVGNNCross, make_paired_loader, mvgnn_train_epoch, mvgnn_eval

SEED = 42
BATCH = 64
MAX_EPOCHS = 60
PATIENCE = 12
LR = 5e-4

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
STAGE1_DIR = os.path.join(os.path.dirname(__file__), 'results', 'stage1_folds')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'stage2_folds')
os.makedirs(OUT_DIR, exist_ok=True)


def main(fold):
    torch.manual_seed(SEED)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')

    ckpt_path = os.path.join(STAGE1_DIR, f'fold_{fold}.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Missing {ckpt_path} -- run train_stage1_fold.py {fold} first.')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    print(f'Fold {fold}: loaded Stage-1 checkpoint, config={cfg}')

    fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
    atom_tr = torch.load(os.path.join(fold_dir, 'atom_train.pt'), weights_only=False)
    bond_tr = torch.load(os.path.join(fold_dir, 'bond_train.pt'), weights_only=False)
    atom_va = torch.load(os.path.join(fold_dir, 'atom_val.pt'), weights_only=False)
    bond_va = torch.load(os.path.join(fold_dir, 'bond_val.pt'), weights_only=False)
    y_tr = np.load(os.path.join(fold_dir, 'y_train.npy'))

    atom_ncv = torch.load(os.path.join(DATA_DIR, 'atom_ncv.pt'), weights_only=False)
    bond_ncv = torch.load(os.path.join(DATA_DIR, 'bond_ncv.pt'), weights_only=False)
    y_ncv = np.load(os.path.join(DATA_DIR, 'y_ncv.npy'))

    atom_test = torch.load(os.path.join(DATA_DIR, 'atom_test.pt'), weights_only=False)
    bond_test = torch.load(os.path.join(DATA_DIR, 'bond_test.pt'), weights_only=False)

    # Stage-2 train = this fold's clin_train + NCV-888 -- simple list-concat,
    # same as DILI_proxy_clean's train_mvgnn_stage2.py.
    atom_aug = atom_tr + atom_ncv
    bond_aug = bond_tr + bond_ncv
    y_aug = np.concatenate([y_tr, y_ncv])
    print(f'Fold {fold} Stage-2 train (clin_train + NCV-888): {len(atom_aug):,}')

    aug_loader = make_paired_loader(atom_aug, bond_aug, BATCH, shuffle=True)
    val_loader = make_paired_loader(atom_va, bond_va, BATCH, shuffle=False)
    test_loader = make_paired_loader(atom_test, bond_test, BATCH, shuffle=False)

    pos_weight = torch.tensor([(y_aug == 0).sum() / (y_aug == 1).sum()], device=DEVICE)
    print(f'pos_weight (Stage-2): {pos_weight.item():.3f}')

    model = MVGNNCross(ckpt['atom_dim'], ckpt['atom_edge_dim'], ckpt['bond_dim'], ckpt['bond_edge_dim'],
                        hidden=cfg['hidden'], n_layers=cfg['n_layers'], dropout=cfg['dropout']).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=cfg['wd'])

    best_auc, best_state, no_improv = 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        mvgnn_train_epoch(model, aug_loader, opt, pos_weight, cfg['lam'], DEVICE)
        _, _, va_auc = mvgnn_eval(model, val_loader, DEVICE)
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = copy.deepcopy(model.state_dict())
            no_improv = 0
        else:
            no_improv += 1
        if epoch % 10 == 0 or no_improv >= PATIENCE:
            print(f'  ep {epoch:3d}  val AUC={va_auc:.4f}  best={best_auc:.4f}')
        if no_improv >= PATIENCE:
            print(f'  early stop @ ep {epoch}')
            break

    print(f'>> Fold {fold} Stage-2 best val AUC={best_auc:.4f}')

    model.load_state_dict(best_state)
    prob_test, y_test, _ = mvgnn_eval(model, test_loader, DEVICE)

    torch.save({
        'state_dict': best_state, 'config': cfg,
        'atom_dim': ckpt['atom_dim'], 'atom_edge_dim': ckpt['atom_edge_dim'],
        'bond_dim': ckpt['bond_dim'], 'bond_edge_dim': ckpt['bond_edge_dim'],
        'fold': fold, 'val_auc': best_auc,
    }, os.path.join(OUT_DIR, f'fold_{fold}.pt'))

    np.savez(os.path.join(OUT_DIR, f'fold_{fold}.npz'),
              val_auc=best_auc, test_probs=prob_test, y_test=y_test)
    print(f'Saved -> {OUT_DIR}/fold_{fold}.pt, fold_{fold}.npz')


if __name__ == '__main__':
    if len(sys.argv) != 2 or not sys.argv[1].isdigit() or not (0 <= int(sys.argv[1]) < 5):
        print(f'Usage: python {sys.argv[0]} <fold_idx 0-4>')
        sys.exit(1)
    main(int(sys.argv[1]))
