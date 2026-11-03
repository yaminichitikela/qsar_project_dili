"""Stage-1 ("Proxy-only") official run for ONE fold: retrains the single
WINNING config (from select_stage1_config.py) on that fold's clin_train
alone, early-stopping on that fold's clin_val -- and this time SAVES the
model checkpoint, which the search-phase config_<i>.slurm jobs never did
(they only kept scores, since 5/6 configs lose and checkpointing all 30
search runs would be wasted work).

held-out-223 predictions are computed here for reference/ensembling in
aggregate_stage1_cv.py; they play no role in training or model selection.

Requires prepare_dili_cv_data.py AND select_stage1_config.py to have run
first.

Usage: python train_stage1_fold.py <fold_idx 0-4>
Outputs (./results/stage1_folds/):
    fold_<i>.pt  -- checkpoint (state_dict + config + dims), for Stage-2 to load
    fold_<i>.npz -- val_auc, test_probs
"""
import copy
import os
import sys

import numpy as np
import torch

from mvgnn_model import MVGNNCross, MAX_EPOCHS, PATIENCE, make_paired_loader, mvgnn_train_epoch, mvgnn_eval

SEED = 42
BATCH = 64

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'stage1_folds')
os.makedirs(OUT_DIR, exist_ok=True)


def main(fold):
    torch.manual_seed(SEED)

    import json
    with open(os.path.join(os.path.dirname(__file__), 'results', 'stage1_winning_config.json')) as f:
        winner = json.load(f)
    cfg = winner['config']
    print(f'Fold {fold}: winning config (idx {winner["config_idx"]}) = {cfg}')

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')

    atom_test = torch.load(os.path.join(DATA_DIR, 'atom_test.pt'), weights_only=False)
    bond_test = torch.load(os.path.join(DATA_DIR, 'bond_test.pt'), weights_only=False)
    test_loader = make_paired_loader(atom_test, bond_test, BATCH, shuffle=False)

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

    best_auc, no_improv, best_state = 0, 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        mvgnn_train_epoch(model, train_loader, opt, pos_weight, cfg['lam'], DEVICE)
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

    print(f'>> Fold {fold} Stage-1 best val AUC={best_auc:.4f}')

    model.load_state_dict(best_state)
    prob_test, y_test, _ = mvgnn_eval(model, test_loader, DEVICE)

    torch.save({
        'state_dict': best_state, 'config': cfg,
        'atom_dim': ATOM_DIM, 'atom_edge_dim': ATOM_EDGE_DIM,
        'bond_dim': BOND_DIM, 'bond_edge_dim': BOND_EDGE_DIM,
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
