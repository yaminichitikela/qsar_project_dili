"""MV-GNN^cross Stage-1 ("Proxy-only" / "Clinical-only" in Table 6's naming):
trains MV-GNN^cross from scratch on clin_train, early-stops on clin_val,
evaluates on held-out-223 -- exactly the same data/role/epochs/patience/lr as
Graphormer/GraphGPS's Stage-1 in proxy_to_dili_consistent copy 2.ipynb cell 17/18
(train_gt_model(..., clin_tr_ld, clin_va_ld, crit_clin, max_epochs=80,
patience=15, lr=1e-3)), so MV-GNN's Stage-1 number is directly comparable to
the existing RF/XGB/Graphormer/GraphGPS Stage-1 rows in that table.

Uses one fixed hyperparameter config (no grid search), matching how
Graphormer/GraphGPS are trained in that notebook (no HP search there either):
hidden=256, n_layers=3, dropout=0.3, lam=0.5, wd=1e-5 -- the same winning
config already found for MV-GNN's own DILI single-split model
(Rebuttal/DILI_mvgnn/modeling/wahab_jobs/results/mvgnn_model.pt).

Requires prepare_proxy_data.py to have been run first (loads from ./data/).
Usage: python train_mvgnn_stage1.py
Outputs (./results/): mvgnn_stage1_results.csv, mvgnn_stage1_test_probs.npy,
                       mvgnn_stage1.pt (checkpoint, reused by Stage-2)
"""
import copy
import os

import numpy as np
import pandas as pd
import torch

from eval_utils import evaluate, METRIC_COLS
from mvgnn_model import MVGNNCross, make_paired_loader, mvgnn_train_epoch, mvgnn_eval

SEED = 42
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)

# Same fixed config as the winning DILI single-split MV-GNN model.
CFG = {'hidden': 256, 'n_layers': 3, 'dropout': 0.3, 'lr': 1e-3, 'lam': 0.5, 'wd': 1e-5}
# Same Stage-1 schedule as Graphormer/GraphGPS in proxy_to_dili_consistent copy 2.ipynb.
MAX_EPOCHS = 80
PATIENCE = 15
BATCH = 64


def main():
    atom_train = torch.load(os.path.join(DATA_DIR, 'atom_train.pt'), weights_only=False)
    atom_val = torch.load(os.path.join(DATA_DIR, 'atom_val.pt'), weights_only=False)
    atom_test = torch.load(os.path.join(DATA_DIR, 'atom_test.pt'), weights_only=False)
    bond_train = torch.load(os.path.join(DATA_DIR, 'bond_train.pt'), weights_only=False)
    bond_val = torch.load(os.path.join(DATA_DIR, 'bond_val.pt'), weights_only=False)
    bond_test = torch.load(os.path.join(DATA_DIR, 'bond_test.pt'), weights_only=False)
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))

    ATOM_DIM = atom_train[0].x.shape[1]
    ATOM_EDGE_DIM = atom_train[0].edge_attr.shape[1]
    BOND_DIM = bond_train[0].x.shape[1]
    BOND_EDGE_DIM = bond_train[0].edge_attr.shape[1]
    print(f'Atom graph: node_dim={ATOM_DIM} edge_dim={ATOM_EDGE_DIM}')
    print(f'Bond graph: node_dim={BOND_DIM} edge_dim={BOND_EDGE_DIM}')

    train_loader = make_paired_loader(atom_train, bond_train, BATCH, shuffle=True)
    val_loader = make_paired_loader(atom_val, bond_val, BATCH, shuffle=False)
    test_loader = make_paired_loader(atom_test, bond_test, BATCH, shuffle=False)

    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], device=DEVICE)
    print(f'Train: {len(atom_train)}  Val: {len(atom_val)}  Test: {len(atom_test)}')
    print(f'pos_weight: {pos_weight.item():.3f}')

    model = MVGNNCross(ATOM_DIM, ATOM_EDGE_DIM, BOND_DIM, BOND_EDGE_DIM,
                        hidden=CFG['hidden'], n_layers=CFG['n_layers'], dropout=CFG['dropout']).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])

    best_auc, best_state, no_improv = 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        mvgnn_train_epoch(model, train_loader, opt, pos_weight, CFG['lam'], DEVICE)
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

    print(f'>> Stage-1 best val AUC={best_auc:.4f}')

    model.load_state_dict(best_state)
    prob_val, y_val, _ = mvgnn_eval(model, val_loader, DEVICE)
    prob_test, y_test, _ = mvgnn_eval(model, test_loader, DEVICE)

    rows = []
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_val, prob_val, 'MV-GNN Stage-1 (Clinical)', cpt), 'Split': 'Val'})
        rows.append({**evaluate(y_test, prob_test, 'MV-GNN Stage-1 (Clinical)', cpt), 'Split': 'Test'})

    df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
    df.to_csv(os.path.join(OUT_DIR, 'mvgnn_stage1_results.csv'), index=False)
    np.save(os.path.join(OUT_DIR, 'mvgnn_stage1_test_probs.npy'), prob_test)
    torch.save({'state_dict': best_state, 'config': CFG,
                'atom_dim': ATOM_DIM, 'atom_edge_dim': ATOM_EDGE_DIM,
                'bond_dim': BOND_DIM, 'bond_edge_dim': BOND_EDGE_DIM},
               os.path.join(OUT_DIR, 'mvgnn_stage1.pt'))
    print(df)
    print(f'Saved -> {OUT_DIR}/mvgnn_stage1_results.csv, mvgnn_stage1_test_probs.npy, mvgnn_stage1.pt')


if __name__ == '__main__':
    main()
