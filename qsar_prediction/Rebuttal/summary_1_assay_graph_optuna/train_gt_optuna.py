"""Graph Transformers (Graphormer/GraphGPS) hyperparameter search via Optuna
(TPE sampler + median pruning), for ONE split unit (one repeat of the random
split, one LOO fold, or one repeat of the scaffold split).

This replaces the fixed 3-config GT_CONFIGS grid used in
Summary_1_models/modeling/wahab_jobs/train_gt.py / train_gt_loo.py /
train_gt_scaffold.py. Same fairness rationale and same search setup as
train_gnn_optuna.py in this folder (ported from the already-validated
DILI_proxy_clean/modeling/wahab_jobs/train_gt_stage1.py search): every split
unit gets its own independent 25-trial Optuna search.

Usage: python train_gt_optuna.py <Graphormer|GraphGPS> <repeat|loo|scaffold> <idx> [n_trials]
    repeat idx: 0-4   (random-split repeat, needs prepare_split_repeat.py + prepare_gt_pe_repeat.py)
    loo idx:    0-9   (LOO fold, needs prepare_loo_folds.py + prepare_gt_pe_loo.py)
    scaffold idx: 0-4 (scaffold-split repeat, needs prepare_scaffold_splits.py + prepare_gt_pe_scaffold.py)

Outputs (./results_<split>/):
    probs/{arch}_{idx}_test_probs.npy   -- needed by aggregate_optuna.py
    {arch}_{idx}_results.csv
    {arch}_{idx}_optuna_trials.csv
    {arch}_{idx}.pt
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
from gt_model import GraphormerModel, GraphGPSModel, ATOM_DIM, PE_DIM, gt_train, gt_eval

SEED = 42
MAX_EPOCHS = 80
PATIENCE = 15
BATCH = 64
N_TRIALS_DEFAULT = 25

ARCHS = ('Graphormer', 'GraphGPS')
SPLITS = {
    'repeat': {'dir': 'data_repeat', 'unit': 'repeat', 'n_units': 5},
    'loo': {'dir': 'data_loo', 'unit': 'fold', 'n_units': 10},
    'scaffold': {'dir': 'data_scaffold', 'unit': 'repeat', 'n_units': 5},
}

# (hidden, heads) pairs kept divisible -- same choices as DILI's train_gt_stage1.py
HIDDEN_HEADS_CHOICES = ['128_4', '256_4', '384_4', '256_8', '384_8', '512_8']

HERE = os.path.dirname(__file__)


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


def main(arch, split, idx, n_trials):
    torch.manual_seed(SEED)
    has_pe = (arch == 'GraphGPS')
    suffix = '_gps' if has_pe else ''
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = SPLITS[split]
    unit_dir_name = f"{cfg['unit']}_{idx}"
    print(f'Architecture: {arch}  Split: {split} ({unit_dir_name})  Device: {DEVICE}  n_trials={n_trials}')

    data_dir = os.path.join(HERE, cfg['dir'], unit_dir_name)
    out_dir = os.path.join(HERE, f'results_{split}')
    probs_dir = os.path.join(out_dir, 'probs')
    os.makedirs(probs_dir, exist_ok=True)

    g_train = torch.load(os.path.join(data_dir, f'graphs_train{suffix}.pt'), weights_only=False)
    g_val = torch.load(os.path.join(data_dir, f'graphs_val{suffix}.pt'), weights_only=False)
    g_test = torch.load(os.path.join(data_dir, f'graphs_test{suffix}.pt'), weights_only=False)

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

    sampler = optuna.samplers.TPESampler(seed=SEED + idx)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))

    pruned_count = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f'>> Optuna search done ({unit_dir_name}): {len(study.trials)} trials, {pruned_count} pruned, '
          f'best val AUC={study.best_value:.4f}  hh={study.best_params["hidden_heads"]}')

    pd.DataFrame(trial_log).to_csv(os.path.join(out_dir, f'{arch.lower()}_{idx}_optuna_trials.csv'), index=False)

    hh = study.best_params['hidden_heads']
    hidden, heads = (int(v) for v in hh.split('_'))
    best_params = {'hidden': hidden, 'num_heads': heads,
                    'num_layers': study.best_params['num_layers'],
                    'dropout': study.best_params['dropout'],
                    'lr': study.best_params['lr']}

    best_auc, best_state, stopped_ep = train_one_config(
        arch, best_params, ATOM_DIM, train_loader, val_loader, criterion, DEVICE,
        has_pe, MAX_EPOCHS, PATIENCE, trial=None)
    print(f'>> Final retrain {arch} {unit_dir_name}: val AUC={best_auc:.4f}  '
          f'stopped@ep{stopped_ep}  params={best_params}')

    model = build_model(arch, ATOM_DIM, best_params).to(DEVICE)
    model.load_state_dict(best_state)
    prob_val, y_val, _ = gt_eval(model, val_loader, DEVICE, has_pe)
    prob_test, y_test, _ = gt_eval(model, test_loader, DEVICE, has_pe)

    np.save(os.path.join(probs_dir, f'{arch.lower()}_{idx}_test_probs.npy'), prob_test)

    rows = []
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_val, prob_val, arch, cpt), 'Split': 'Val', cfg['unit'].capitalize(): idx})
        rows.append({**evaluate(y_test, prob_test, arch, cpt), 'Split': 'Test', cfg['unit'].capitalize(): idx})

    df = pd.DataFrame(rows)[[cfg['unit'].capitalize(), 'Split'] + METRIC_COLS]
    df.to_csv(os.path.join(out_dir, f'{arch.lower()}_{idx}_results.csv'), index=False)
    torch.save({'state_dict': best_state, 'config': best_params},
               os.path.join(out_dir, f'{arch.lower()}_{idx}.pt'))
    print(df)
    print(f'Saved -> {out_dir}/{arch.lower()}_{idx}_results.csv, probs/{arch.lower()}_{idx}_test_probs.npy')


if __name__ == '__main__':
    if len(sys.argv) not in (4, 5) or sys.argv[1] not in ARCHS or sys.argv[2] not in SPLITS:
        print(f'Usage: python {sys.argv[0]} <{"|".join(ARCHS)}> <repeat|loo|scaffold> <idx> [n_trials]')
        sys.exit(1)
    arch_arg, split_arg, idx_arg = sys.argv[1], sys.argv[2], int(sys.argv[3])
    n_units = SPLITS[split_arg]['n_units']
    if not (0 <= idx_arg < n_units):
        print(f'idx must be 0-{n_units - 1} for split={split_arg}')
        sys.exit(1)
    n_trials_arg = int(sys.argv[4]) if len(sys.argv) == 5 else N_TRIALS_DEFAULT
    main(arch_arg, split_arg, idx_arg, n_trials_arg)
