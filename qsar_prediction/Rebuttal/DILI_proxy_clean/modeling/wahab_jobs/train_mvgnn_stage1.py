"""MV-GNN^cross Stage-1 ("Proxy-only" / "Clinical-only" in Table 6's naming):
Optuna hyperparameter search (TPE sampler + median pruning) on clin_train,
validated on clin_val -- same treatment as GNN/GT Stage-1, replacing the old
fixed 3-config MVGNN_CONFIGS grid. Searches hidden/n_layers/dropout/lr/
weight_decay plus MV-GNN's own disagreement-loss weight (lam).

After the search, the winning trial's hyperparameters are retrained once more
(same schedule) to get clean, freshly-saved weights for Stage-2.

Requires prepare_proxy_data_clean.py to have run first, and `optuna` installed.
Usage: python train_mvgnn_stage1.py [n_trials]
Outputs (./results/): mvgnn_stage1_results.csv, mvgnn_stage1_test_probs.npy,
                       mvgnn_stage1.pt, mvgnn_stage1_optuna_trials.csv
"""
import copy
import os
import sys

import numpy as np
import optuna
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

MAX_EPOCHS = 80
PATIENCE = 15
BATCH = 64
N_TRIALS_DEFAULT = 25


def suggest_params(trial):
    return {
        'hidden': trial.suggest_categorical('hidden', [64, 128, 256, 384]),
        'n_layers': trial.suggest_int('n_layers', 2, 5),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'lam': trial.suggest_float('lam', 0.1, 2.0),
        'wd': trial.suggest_float('wd', 1e-6, 1e-3, log=True),
    }


def train_one_config(params, atom_dim, atom_edge_dim, bond_dim, bond_edge_dim,
                      train_loader, val_loader, pos_weight, device, max_epochs, patience, trial=None):
    model = MVGNNCross(atom_dim, atom_edge_dim, bond_dim, bond_edge_dim,
                        hidden=params['hidden'], n_layers=params['n_layers'], dropout=params['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    best_auc, best_state, no_improv = 0, None, 0
    for epoch in range(1, max_epochs + 1):
        mvgnn_train_epoch(model, train_loader, opt, pos_weight, params['lam'], device)
        _, _, va_auc = mvgnn_eval(model, val_loader, device)
        if va_auc > best_auc:
            best_auc, best_state, no_improv = va_auc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improv += 1

        if trial is not None:
            trial.report(va_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if no_improv >= patience:
            break
    return best_auc, best_state, epoch


def main(n_trials):
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
    print(f'n_trials={n_trials}')

    train_loader = make_paired_loader(atom_train, bond_train, BATCH, shuffle=True)
    val_loader = make_paired_loader(atom_val, bond_val, BATCH, shuffle=False)
    test_loader = make_paired_loader(atom_test, bond_test, BATCH, shuffle=False)

    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()], device=DEVICE)
    print(f'Train: {len(atom_train)}  Val: {len(atom_val)}  Test: {len(atom_test)}')
    print(f'pos_weight: {pos_weight.item():.3f}')

    trial_log = []

    def objective(trial):
        params = suggest_params(trial)
        best_auc, _, stopped_ep = train_one_config(
            params, ATOM_DIM, ATOM_EDGE_DIM, BOND_DIM, BOND_EDGE_DIM,
            train_loader, val_loader, pos_weight, DEVICE, MAX_EPOCHS, PATIENCE, trial=trial)
        trial_log.append({**params, 'val_auc': best_auc, 'stopped_epoch': stopped_ep, 'trial': trial.number})
        return best_auc

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))

    pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f'>> Optuna search done: {len(study.trials)} trials, {pruned_count} pruned, '
          f'best val AUC={study.best_value:.4f}  params={study.best_params}')

    pd.DataFrame(trial_log).to_csv(os.path.join(OUT_DIR, 'mvgnn_stage1_optuna_trials.csv'), index=False)

    best_params = study.best_params
    best_auc, best_state, stopped_ep = train_one_config(
        best_params, ATOM_DIM, ATOM_EDGE_DIM, BOND_DIM, BOND_EDGE_DIM,
        train_loader, val_loader, pos_weight, DEVICE, MAX_EPOCHS, PATIENCE, trial=None)
    print(f'>> Stage-1 final retrain MV-GNN: val AUC={best_auc:.4f}  stopped@ep{stopped_ep}  params={best_params}')

    model = MVGNNCross(ATOM_DIM, ATOM_EDGE_DIM, BOND_DIM, BOND_EDGE_DIM,
                        hidden=best_params['hidden'], n_layers=best_params['n_layers'],
                        dropout=best_params['dropout']).to(DEVICE)
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
    torch.save({'state_dict': best_state, 'config': best_params,
                'atom_dim': ATOM_DIM, 'atom_edge_dim': ATOM_EDGE_DIM,
                'bond_dim': BOND_DIM, 'bond_edge_dim': BOND_EDGE_DIM},
               os.path.join(OUT_DIR, 'mvgnn_stage1.pt'))
    print(df)
    print(f'Saved -> {OUT_DIR}/mvgnn_stage1_results.csv, mvgnn_stage1_optuna_trials.csv')


if __name__ == '__main__':
    n_trials = int(sys.argv[1]) if len(sys.argv) == 2 else N_TRIALS_DEFAULT
    main(n_trials)
