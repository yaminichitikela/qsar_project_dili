"""GNN family (GCN/GAT/GIN/SAGE) Stage-1 ("Proxy-only"): Optuna hyperparameter
search (TPE sampler + median pruning) on clin_train, validated on clin_val,
replacing the old fixed 3-config grid. Pruning compares each trial's val AUC
at a given epoch against the running median of other trials at that same
epoch -- this fixes the failure mode found in proxy_to_dili_consistent
copy 4.ipynb, where a fixed short (25-epoch) probe crowned a hyperparameter
config that then did WORSE once fully trained (Graphormer: hidden=512 "won"
a short search but scored lower than the untuned hidden=256 baseline once
trained to convergence). Median pruning only cuts a trial once it's actually
behind its peers at that point in training, not on a blind epoch cutoff.

After the search, the winning trial's hyperparameters are retrained once
more (same schedule) to get clean, freshly-saved weights for Stage-2.

Usage: python train_gnn_stage1.py <GCN|GAT|GIN|SAGE> [n_trials]
Requires prepare_proxy_data_clean.py to have run first, and `optuna` installed
(pip install optuna) in the Wahab env.
Outputs (./results/): {arch}_stage1_results.csv, {arch}_stage1_test_probs.npy,
                       {arch}_stage1.pt, {arch}_stage1_optuna_trials.csv
"""
import copy
import os
import sys

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from eval_utils import evaluate, METRIC_COLS
from gnn_model import MolGNN, gnn_train_epoch, gnn_eval

SEED = 42
MAX_EPOCHS = 80
PATIENCE = 15
BATCH = 64
N_TRIALS_DEFAULT = 25

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def suggest_params(trial, arch):
    params = {
        'hidden': trial.suggest_categorical('hidden', [64, 128, 256, 384]),
        'n_layers': trial.suggest_int('n_layers', 2, 5),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
    }
    if arch == 'GAT':
        params['heads'] = trial.suggest_categorical('heads', [2, 4, 8])
    return params


def train_one_config(arch, params, atom_dim, train_loader, val_loader, criterion, device,
                      max_epochs, patience, trial=None):
    model = MolGNN(atom_dim, hidden=params['hidden'], n_layers=params['n_layers'],
                    dropout=params['dropout'], conv_type=arch, heads=params.get('heads', 4)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    best_auc, best_state, no_improv = 0, None, 0
    for epoch in range(1, max_epochs + 1):
        gnn_train_epoch(model, train_loader, opt, criterion, device)
        _, _, va_auc = gnn_eval(model, val_loader, device)
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


def main(arch, n_trials):
    torch.manual_seed(SEED)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Architecture: {arch}  Device: {DEVICE}  n_trials={n_trials}')

    g_train = torch.load(os.path.join(DATA_DIR, 'gt_train.pt'), weights_only=False)
    g_val = torch.load(os.path.join(DATA_DIR, 'gt_val.pt'), weights_only=False)
    g_test = torch.load(os.path.join(DATA_DIR, 'gt_test.pt'), weights_only=False)
    ATOM_DIM = g_train[0].x.shape[1]

    train_loader = DataLoader(g_train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(g_val, batch_size=BATCH, shuffle=False)
    test_loader = DataLoader(g_test, batch_size=BATCH, shuffle=False)

    y_train_np = np.array([g.y.item() for g in g_train])
    pos_weight = torch.tensor([(y_train_np == 0).sum() / (y_train_np == 1).sum()], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f'Train: {len(g_train)}  Val: {len(g_val)}  Test: {len(g_test)}  pos_weight={pos_weight.item():.3f}')

    trial_log = []

    def objective(trial):
        params = suggest_params(trial, arch)
        best_auc, _, stopped_ep = train_one_config(
            arch, params, ATOM_DIM, train_loader, val_loader, criterion, DEVICE,
            MAX_EPOCHS, PATIENCE, trial=trial)
        trial_log.append({**params, 'val_auc': best_auc, 'stopped_epoch': stopped_ep,
                           'trial': trial.number, 'pruned': False})
        return best_auc

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))

    pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f'>> Optuna search done: {len(study.trials)} trials, {pruned_count} pruned, '
          f'best val AUC={study.best_value:.4f}  params={study.best_params}')

    pd.DataFrame(trial_log).to_csv(os.path.join(OUT_DIR, f'{arch.lower()}_stage1_optuna_trials.csv'), index=False)

    # Retrain the winning config once more for clean weights.
    best_params = study.best_params
    best_auc, best_state, stopped_ep = train_one_config(
        arch, best_params, ATOM_DIM, train_loader, val_loader, criterion, DEVICE,
        MAX_EPOCHS, PATIENCE, trial=None)
    print(f'>> Stage-1 final retrain {arch}: val AUC={best_auc:.4f}  stopped@ep{stopped_ep}  params={best_params}')

    model = MolGNN(ATOM_DIM, hidden=best_params['hidden'], n_layers=best_params['n_layers'],
                    dropout=best_params['dropout'], conv_type=arch, heads=best_params.get('heads', 4)).to(DEVICE)
    model.load_state_dict(best_state)
    prob_val, y_val, _ = gnn_eval(model, val_loader, DEVICE)
    prob_test, y_test, _ = gnn_eval(model, test_loader, DEVICE)

    rows = []
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_val, prob_val, f'{arch} Stage-1 (Proxy-only)', cpt), 'Split': 'Val'})
        rows.append({**evaluate(y_test, prob_test, f'{arch} Stage-1 (Proxy-only)', cpt), 'Split': 'Test'})

    df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
    df.to_csv(os.path.join(OUT_DIR, f'{arch.lower()}_stage1_results.csv'), index=False)
    np.save(os.path.join(OUT_DIR, f'{arch.lower()}_stage1_test_probs.npy'), prob_test)
    torch.save({'state_dict': best_state, 'config': best_params, 'atom_dim': ATOM_DIM},
               os.path.join(OUT_DIR, f'{arch.lower()}_stage1.pt'))
    print(df)
    print(f'Saved -> {OUT_DIR}/{arch.lower()}_stage1_results.csv, {arch.lower()}_stage1_optuna_trials.csv')


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ('GCN', 'GAT', 'GIN', 'SAGE'):
        print(f'Usage: python {sys.argv[0]} <GCN|GAT|GIN|SAGE> [n_trials]')
        sys.exit(1)
    n_trials = int(sys.argv[2]) if len(sys.argv) == 3 else N_TRIALS_DEFAULT
    main(sys.argv[1], n_trials)
