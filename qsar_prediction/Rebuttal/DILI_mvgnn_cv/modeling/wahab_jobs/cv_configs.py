"""Shared hyperparameter search space for the DILI MV-GNN^cross CV search.
Kept in its own module (not inside train_mvgnn_cv_config.py) so that
select_best_and_ensemble.py can import CONFIGS without triggering any
training code as an import side effect."""

CONFIGS = [
    {'hidden': 128, 'n_layers': 3, 'dropout': 0.3, 'lr': 1e-3, 'lam': 1.0, 'wd': 1e-5},
    {'hidden': 256, 'n_layers': 3, 'dropout': 0.3, 'lr': 1e-3, 'lam': 0.5, 'wd': 1e-5},
    {'hidden': 256, 'n_layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'lam': 0.5, 'wd': 1e-4},
    {'hidden': 384, 'n_layers': 3, 'dropout': 0.4, 'lr': 5e-4, 'lam': 1.0, 'wd': 1e-4},
    {'hidden': 128, 'n_layers': 4, 'dropout': 0.3, 'lr': 5e-4, 'lam': 1.5, 'wd': 1e-5},
    {'hidden': 256, 'n_layers': 3, 'dropout': 0.5, 'lr': 2e-4, 'lam': 0.3, 'wd': 1e-4},
]
