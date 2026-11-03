"""Runs 5-fold CV for ONE candidate hyperparameter config (by index into
CONFIGS below), so all configs can run as separate parallel SLURM jobs
instead of one long sequential job. This is the Stage-1 SEARCH phase only --
see select_stage1_config.py for how the results from all 6 configs get
combined to pick the winner, and train_stage1_fold.py for how the winning
config gets retrained once per fold WITH checkpoints saved (this script
never saves checkpoints, since 5/6 configs lose and checkpointing all 30
search runs would be wasted work).

held-out-223 predictions are computed here (once per fold) purely for
convenience -- they play NO role in which config wins (that's decided by
select_stage1_config.py using only fold_val_aucs, i.e. the 5-fold CV score)
or in this script's own early stopping (each fold uses only its own
held-out 1/5 for that).

Usage: python train_mvgnn_cv_config.py <config_idx 0-5>
Requires prepare_dili_cv_data.py to have been run first (loads from ./data/).
"""
import copy
import os
import sys

import numpy as np
import torch

from mvgnn_model import MVGNNCross, MAX_EPOCHS, PATIENCE, make_paired_loader, mvgnn_train_epoch, mvgnn_eval
from cv_configs import CONFIGS

SEED = 42
N_FOLDS = 5
BATCH = 64


def main(cfg_idx):
    torch.manual_seed(SEED)
    cfg = CONFIGS[cfg_idx]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Config {cfg_idx}: {cfg}  Device: {DEVICE}')

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'cv_search')
    os.makedirs(OUT_DIR, exist_ok=True)

    atom_test = torch.load(os.path.join(DATA_DIR, 'atom_test.pt'), weights_only=False)
    bond_test = torch.load(os.path.join(DATA_DIR, 'bond_test.pt'), weights_only=False)
    test_loader = make_paired_loader(atom_test, bond_test, BATCH, shuffle=False)

    fold_val_aucs = []
    fold_test_probs = []

    for fold in range(N_FOLDS):
        fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
        atom_tr = torch.load(os.path.join(fold_dir, 'atom_train.pt'), weights_only=False)
        bond_tr = torch.load(os.path.join(fold_dir, 'bond_train.pt'), weights_only=False)
        atom_va = torch.load(os.path.join(fold_dir, 'atom_val.pt'), weights_only=False)
        bond_va = torch.load(os.path.join(fold_dir, 'bond_val.pt'), weights_only=False)
        y_tr = np.load(os.path.join(fold_dir, 'y_train.npy'))

        ATOM_DIM = atom_tr[0].x.shape[1]
        ATOM_EDGE_DIM = atom_tr[0].edge_attr.shape[1]
        BOND_DIM = bond_tr[0].x.shape[1]
        BOND_EDGE_DIM = bond_tr[0].edge_attr.shape[1]

        train_loader = make_paired_loader(atom_tr, bond_tr, BATCH, shuffle=True)
        val_loader = make_paired_loader(atom_va, bond_va, BATCH, shuffle=False)
        pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()], device=DEVICE)

        model = MVGNNCross(ATOM_DIM, ATOM_EDGE_DIM, BOND_DIM, BOND_EDGE_DIM,
                            hidden=cfg['hidden'], n_layers=cfg['n_layers'], dropout=cfg['dropout']).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])

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

        print(f'  fold {fold}: val AUC={best_ep_auc:.4f}  stopped@ep{epoch}')
        fold_val_aucs.append(best_ep_auc)

        model.load_state_dict(best_ep_state)
        prob_test, _, _ = mvgnn_eval(model, test_loader, DEVICE)
        fold_test_probs.append(prob_test)

    mean_cv_auc = float(np.mean(fold_val_aucs))
    std_cv_auc = float(np.std(fold_val_aucs))
    print(f'>> Config {cfg_idx} CV val AUC = {mean_cv_auc:.4f} +/- {std_cv_auc:.4f}')

    np.savez(os.path.join(OUT_DIR, f'config_{cfg_idx}.npz'),
             cfg=cfg, fold_val_aucs=fold_val_aucs, fold_test_probs=np.array(fold_test_probs),
             mean_cv_auc=mean_cv_auc, std_cv_auc=std_cv_auc)
    print(f'Saved -> {OUT_DIR}/config_{cfg_idx}.npz')


if __name__ == '__main__':
    if len(sys.argv) != 2 or not sys.argv[1].isdigit() or not (0 <= int(sys.argv[1]) < len(CONFIGS)):
        print(f'Usage: python {sys.argv[0]} <config_idx 0-{len(CONFIGS)-1}>')
        sys.exit(1)
    main(int(sys.argv[1]))
