"""MV-GNN^cross Stage-2 ("Proxy+DILI" / "Clinical+NCV-888" in Table 6's
naming): loads the Stage-1 checkpoint, fine-tunes on clin_train + NCV-888
(~11,194 compounds) as one combined training set, early-stops on the same
clin_val, evaluates on held-out-223 -- exactly the same data/role/epochs/
patience/lr as Graphormer/GraphGPS's Stage-2 in proxy_to_dili_consistent
copy 2.ipynb cell 17/18 (train_gt_model(..., aug_tr_ld, clin_va_ld, crit_aug,
max_epochs=60, patience=12, lr=5e-4), loading Stage-1 weights into a fresh
model first) -- no gradual unfreezing, all parameters fine-tuned together,
matching that notebook's actual (simpler) Stage-2 procedure.

Requires prepare_proxy_data.py AND train_mvgnn_stage1.py to have run first.
Usage: python train_mvgnn_stage2.py
Outputs (./results/): mvgnn_stage2_results.csv, mvgnn_stage2_test_probs.npy
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

# Same Stage-2 schedule as Graphormer/GraphGPS.
MAX_EPOCHS = 60
PATIENCE = 12
LR = 5e-4
BATCH = 64


def main():
    ckpt_path = os.path.join(OUT_DIR, 'mvgnn_stage1.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Missing {ckpt_path} -- run train_mvgnn_stage1.py first.')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    print(f'Loaded Stage-1 checkpoint, config={cfg}')

    atom_train = torch.load(os.path.join(DATA_DIR, 'atom_train.pt'), weights_only=False)
    atom_val = torch.load(os.path.join(DATA_DIR, 'atom_val.pt'), weights_only=False)
    atom_ncv = torch.load(os.path.join(DATA_DIR, 'atom_ncv.pt'), weights_only=False)
    atom_test = torch.load(os.path.join(DATA_DIR, 'atom_test.pt'), weights_only=False)
    bond_train = torch.load(os.path.join(DATA_DIR, 'bond_train.pt'), weights_only=False)
    bond_val = torch.load(os.path.join(DATA_DIR, 'bond_val.pt'), weights_only=False)
    bond_ncv = torch.load(os.path.join(DATA_DIR, 'bond_ncv.pt'), weights_only=False)
    bond_test = torch.load(os.path.join(DATA_DIR, 'bond_test.pt'), weights_only=False)
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_ncv = np.load(os.path.join(DATA_DIR, 'y_ncv.npy'))

    # Stage-2 training set = clin_train + NCV-888, same simple list-concat as
    # the notebook's `g_aug_tr = g_c_tr + ncv_graphs`.
    atom_aug = atom_train + atom_ncv
    bond_aug = bond_train + bond_ncv
    y_aug = np.concatenate([y_train, y_ncv])
    print(f'Stage-2 train (clin_train + NCV-888): {len(atom_aug):,}')

    aug_loader = make_paired_loader(atom_aug, bond_aug, BATCH, shuffle=True)
    val_loader = make_paired_loader(atom_val, bond_val, BATCH, shuffle=False)
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

    print(f'>> Stage-2 best val AUC={best_auc:.4f}')

    model.load_state_dict(best_state)
    prob_val, y_val, _ = mvgnn_eval(model, val_loader, DEVICE)
    prob_test, y_test, _ = mvgnn_eval(model, test_loader, DEVICE)

    rows = []
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_val, prob_val, 'MV-GNN Stage-2 (Clinical+NCV)', cpt), 'Split': 'Val'})
        rows.append({**evaluate(y_test, prob_test, 'MV-GNN Stage-2 (Clinical+NCV)', cpt), 'Split': 'Test'})

    df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
    df.to_csv(os.path.join(OUT_DIR, 'mvgnn_stage2_results.csv'), index=False)
    np.save(os.path.join(OUT_DIR, 'mvgnn_stage2_test_probs.npy'), prob_test)
    print(df)
    print(f'Saved -> {OUT_DIR}/mvgnn_stage2_results.csv, mvgnn_stage2_test_probs.npy')


if __name__ == '__main__':
    main()
