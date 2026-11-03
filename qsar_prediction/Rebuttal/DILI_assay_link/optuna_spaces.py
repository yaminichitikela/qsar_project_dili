"""Optuna search spaces for all ten model families, shared by the assay side and
the DILI side of the linking experiment.

WHY OPTUNA FOR EVERY FAMILY
---------------------------
The older pipelines gave the families unequal search budgets: RF/SVM/XGBoost got
a full grid (9-16 configurations, re-searched inside every fold), while the seven
graph models got three hand-picked configurations. A reviewer can fairly ask
whether the graph models simply had a cheaper search. Here every family gets its
own independent search of comparable budget, selected on validation ROC-AUC, so
no family is advantaged by the protocol.

Budgets: N_TRIALS_GRAPH = 25 for the graph families (matching
DILI_proxy_clean/train_gnn_stage1.py and summary_1_assay_graph_optuna), and
N_TRIALS_CLASSICAL = 50 for RF/SVM/XGBoost, which is affordable on CPU and more
thorough than the old grid.

The graph search spaces are the ones already validated in
DILI_proxy_clean/modeling/wahab_jobs/train_gnn_stage1.py and train_gt_stage1.py,
extended to cover the two architectures this folder corrects (true Graphormer and
sign-flipped GraphGPS).
"""
N_TRIALS_GRAPH = 25
N_TRIALS_CLASSICAL = 50

GRAPH_FAMILIES = ['GCN', 'GAT', 'GIN', 'SAGE', 'Graphormer', 'GraphGPS', 'MVGNN']
CLASSICAL_FAMILIES = ['RF', 'SVM', 'XGB']
ALL_FAMILIES = CLASSICAL_FAMILIES + GRAPH_FAMILIES   # the 10-model consensus

# hidden must stay divisible by the head count, so they are suggested as a pair
HIDDEN_HEADS_CHOICES = ['128_4', '256_4', '384_4', '256_8', '384_8', '512_8']


def suggest(trial, family):
    """Returns a hyperparameter dict for one Optuna trial of `family`."""
    if family in ('GCN', 'GIN', 'SAGE', 'GAT'):
        p = {
            'hidden': trial.suggest_categorical('hidden', [64, 128, 256, 384]),
            'n_layers': trial.suggest_int('n_layers', 2, 5),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        }
        if family == 'GAT':
            p['heads'] = trial.suggest_categorical('heads', [2, 4, 8])
        return p

    if family in ('Graphormer', 'GraphGPS'):
        hh = trial.suggest_categorical('hidden_heads', HIDDEN_HEADS_CHOICES)
        hidden, heads = (int(v) for v in hh.split('_'))
        p = {
            'hidden': hidden,
            'num_heads': heads,
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'lr': trial.suggest_float('lr', 1e-4, 2e-3, log=True),
        }
        if family == 'Graphormer':
            # true Graphormer benefits from weight decay at these depths
            p['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        return p

    if family == 'MVGNN':
        return {
            'hidden': trial.suggest_categorical('hidden', [64, 128, 256, 384]),
            'n_layers': trial.suggest_int('n_layers', 2, 5),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
            'lam': trial.suggest_float('lam', 0.1, 2.0),      # disagreement loss weight
            'wd': trial.suggest_float('wd', 1e-6, 1e-3, log=True),
        }

    if family == 'RF':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.1, 0.3]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'max_depth': trial.suggest_categorical('max_depth', [None, 10, 20, 40]),
        }

    if family == 'SVM':
        return {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto', 1e-2, 1e-3, 1e-4]),
        }

    if family == 'XGB':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }

    raise ValueError(f'Unknown family: {family}')


def n_trials(family):
    return N_TRIALS_CLASSICAL if family in CLASSICAL_FAMILIES else N_TRIALS_GRAPH
