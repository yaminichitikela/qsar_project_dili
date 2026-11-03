"""Graph Transformers (Graphormer/GraphGPS) Stage-1 ("Proxy-only"): Optuna
hyperparameter search (TPE sampler + median pruning) on clin_train, validated
on clin_val, replacing the old fixed 3-config grid AND copy 4.ipynb's flawed
12-config search (which ranked configs on a short 25-epoch/patience=8 probe --
its "winning" wider Graphormer config, hidden=512, then scored WORSE once
fully trained than the untuned hidden=256 baseline; see prep notes). Median
pruning compares each trial's val AUC against the running median of other
trials at the same epoch, so a trial is only cut once it's actually behind
its peers at that point -- not on a blind short cutoff.

After the search, the winning trial's hyperparameters are retrained once more
(same schedule) to get clean, freshly-saved weights for Stage-2.

Usage: python train_gt_stage1.py <Graphormer|GraphGPS> [n_trials]
Requires prepare_proxy_data_clean.py to have run first, and `optuna` installed.
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
from gt_model import GraphormerModel, GraphGPSModel, PE_DIM, gt_train, gt_eval

SEED = 42
MAX_EPOCHS = 80
PATIENCE = 15
BATCH = 64
N_TRIALS_DEFAULT = 25

# (hidden, heads) pairs kept divisible -- same style as GT_CONFIGS elsewhere.
HIDDEN_HEADS_CHOICES = ['128_4', '256_4', '384_4', '256_8', '384_8', '512_8']

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def suggest_params(trial):
    hh = trial.suggest_categorical('hidden_heads', HIDDEN_HEADS_CHOICES)
    hidden, heads = (int(v) for v in hh.split('_'))
    return {
        'hidden': hidden,
        'num_heads': heads,
        'num_layers': trial.suggest_int('num_layers', 2, 6),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        'lr': trial.suggest_float('lr', 1e-4, 2e-3, log=True),
    }


def build_model(arch, atom_dim, cfg):
    if arch == 'Graphormer':
        return GraphormerModel(atom_dim, cfg['hidden'], cfg['num_heads'], cfg['num_layers'], cfg['dropout'])
    return GraphGPSModel(atom_dim, cfg['hidden'], PE_DIM, cfg['num_heads'], cfg['num_layers'], cfg['dropout'])


def train_one_config(arch, params, atom_dim, train_loader, val_loader, criterion, device,
                      has_pe, max_epochs, patience, trial=None):
    model = build_model(arch, atom_dim, params).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'])

    best_auc, best_state, no_improv = 0, None, 0
    for epoch in range(1, max_epochs + 1):
        gt_train(model, opt, train_loader, criterion, device, has_pe)
        _, _, va_auc = gt_eval(model, val_loader, device, has_pe)
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
    has_pe = (arch == 'GraphGPS')
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
        params = suggest_params(trial)
        best_auc, _, stopped_ep = train_one_config(
            arch, params, ATOM_DIM, train_loader, val_loader, criterion, DEVICE,
            has_pe, MAX_EPOCHS, PATIENCE, trial=trial)
        trial_log.append({**params, 'val_auc': best_auc, 'stopped_epoch': stopped_ep, 'trial': trial.number})
        return best_auc

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))

    pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f'>> Optuna search done: {len(study.trials)} trials, {pruned_count} pruned, '
          f'best val AUC={study.best_value:.4f}  hh={study.best_params["hidden_heads"]}')

    pd.DataFrame(trial_log).to_csv(os.path.join(OUT_DIR, f'{arch.lower()}_stage1_optuna_trials.csv'), index=False)

    hh = study.best_params['hidden_heads']
    hidden, heads = (int(v) for v in hh.split('_'))
    best_params = {'hidden': hidden, 'num_heads': heads,
                    'num_layers': study.best_params['num_layers'],
                    'dropout': study.best_params['dropout'],
                    'lr': study.best_params['lr']}

    best_auc, best_state, stopped_ep = train_one_config(
        arch, best_params, ATOM_DIM, train_loader, val_loader, criterion, DEVICE,
        has_pe, MAX_EPOCHS, PATIENCE, trial=None)
    print(f'>> Stage-1 final retrain {arch}: val AUC={best_auc:.4f}  stopped@ep{stopped_ep}  params={best_params}')

    model = build_model(arch, ATOM_DIM, best_params).to(DEVICE)
    model.load_state_dict(best_state)
    prob_val, y_val, _ = gt_eval(model, val_loader, DEVICE, has_pe)
    prob_test, y_test, _ = gt_eval(model, test_loader, DEVICE, has_pe)

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
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ('Graphormer', 'GraphGPS'):
        print(f'Usage: python {sys.argv[0]} <Graphormer|GraphGPS> [n_trials]')
        sys.exit(1)
    n_trials = int(sys.argv[2]) if len(sys.argv) == 3 else N_TRIALS_DEFAULT
    main(sys.argv[1], n_trials)
