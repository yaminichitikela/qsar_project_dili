"""Classical models (RF/SVM/XGB) full CV pipeline for ONE (model, stage):
grid search (RF/XGB: 8 configs, SVM: 6 configs), each config's mean 5-fold CV
val AUC decides the winner (same selection principle as the GNN/GT/MV-GNN CV
pipelines -- one hyperparameter decision, not per-fold decisions), then the
winning config's 5 already-fitted fold models are ensembled on held-out-223.

Unlike the graph models, this doesn't need separate search/select/retrain
phases: classical fits are cheap and (for RF/XGB) exactly reproducible given
a fixed random_state, so the search loop's own fitted models ARE the
official per-fold models -- no wasted-checkpoint concern, no need to retrain.
SVM is the one exception: search uses decision_function() (ROC-AUC-invariant,
avoids paying for probability=True's internal Platt-scaling calibration on
every config); only once the winning config is known does each fold get
refit with probability=True, exactly once per fold (5 total, not 5xN_configs).

Stage-1 ("Proxy-only"): 5-fold CV over the clinical pool alone.
Stage-2 ("Proxy+DILI"): independent re-search on clin_train_fold + NCV-888
per fold (classical models have no weights to fine-tune from, same reasoning
as DILI_proxy_clean's classical Stage-2).

Usage: python train_classical_cv.py <RF|SVM|XGB> <1|2>
Requires prepare_dili_cv_data.py to have been run first.
Outputs (./results/): classical_cv_stage<N>_<model>_results.csv,
                       classical_cv_stage<N>_<model>_ensemble_test_probs.npy
"""
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from classical_config import RF_GRID, XGB_GRID, SVM_GRID, make_rf, make_xgb, make_svm
from eval_utils import evaluate, METRIC_COLS

MODELS = ('RF', 'SVM', 'XGB')
N_FOLDS = 5

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def load_fold(fold, stage):
    fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
    X_tr = np.load(os.path.join(fold_dir, 'X_train.npy'))
    y_tr = np.load(os.path.join(fold_dir, 'y_train.npy'))
    X_va = np.load(os.path.join(fold_dir, 'X_val.npy'))
    y_va = np.load(os.path.join(fold_dir, 'y_val.npy'))
    if stage == 2:
        X_ncv = np.load(os.path.join(DATA_DIR, 'X_ncv.npy'))
        y_ncv = np.load(os.path.join(DATA_DIR, 'y_ncv.npy'))
        X_tr = np.vstack([X_tr, X_ncv])
        y_tr = np.concatenate([y_tr, y_ncv])
    return X_tr, y_tr, X_va, y_va


def main(model, stage):
    stage_label = 'Proxy-only' if stage == 1 else 'Proxy+DILI'
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    grid = {'RF': RF_GRID, 'XGB': XGB_GRID, 'SVM': SVM_GRID}[model]
    print(f'{model} Stage-{stage} ({stage_label}): {len(grid)}-config grid x {N_FOLDS} folds')

    # ── per-fold data, loaded once ──────────────────────────────────────────
    folds = [load_fold(f, stage) for f in range(N_FOLDS)]

    # ── grid search: for every config, fit on each fold's train, score on
    #    that fold's own val -- mean across folds decides the winner. RF/XGB
    #    fits are kept (cheap, exact); SVM uses decision_function() during
    #    search to avoid paying for probability=True N_configs times. ───────
    config_mean_aucs = []
    config_fold_clfs = []   # only meaningfully cached for RF/XGB
    for cfg in grid:
        fold_aucs = []
        fold_clfs = []
        for X_tr, y_tr, X_va, y_va in folds:
            if model == 'RF':
                clf = make_rf(cfg); clf.fit(X_tr, y_tr)
                auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
            elif model == 'XGB':
                clf = make_xgb(cfg, y_tr); clf.fit(X_tr, y_tr)
                auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
            else:  # SVM -- no calibration during search
                clf = make_svm(cfg, calibrated=False); clf.fit(X_tr, y_tr)
                auc = roc_auc_score(y_va, clf.decision_function(X_va))
            fold_aucs.append(auc)
            fold_clfs.append(clf)
        mean_auc = float(np.mean(fold_aucs))
        config_mean_aucs.append(mean_auc)
        config_fold_clfs.append(fold_clfs)
        print(f'  {cfg}: fold AUCs={[round(a, 4) for a in fold_aucs]}  mean={mean_auc:.4f}')

    best_idx = int(np.argmax(config_mean_aucs))
    best_cfg = grid[best_idx]
    print(f'\n>> WINNING CONFIG: {best_cfg}  (mean CV val AUC = {config_mean_aucs[best_idx]:.4f})')

    search_df = pd.DataFrame([{'ConfigIdx': i, **cfg, 'MeanCV_ValAUC': auc}
                               for i, (cfg, auc) in enumerate(zip(grid, config_mean_aucs))])
    search_df = search_df.sort_values('MeanCV_ValAUC', ascending=False)
    search_df.to_csv(os.path.join(OUT_DIR, f'classical_cv_stage{stage}_{model.lower()}_search.csv'), index=False)

    # ── official per-fold models for the winning config ─────────────────────
    fold_val_aucs, fold_test_probs = [], []
    for i, (X_tr, y_tr, X_va, y_va) in enumerate(folds):
        if model == 'SVM':
            # refit ONCE per fold with probability=True, exactly here (not
            # during search) -- 5 calibrated fits total for the whole script,
            # not 5 x len(SVM_GRID).
            clf = make_svm(best_cfg, calibrated=True)
            clf.fit(X_tr, y_tr)
            val_auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        else:
            clf = config_fold_clfs[best_idx][i]
            val_auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
        fold_val_aucs.append(val_auc)
        fold_test_probs.append(clf.predict_proba(X_test)[:, 1])

    fold_val_aucs = np.array(fold_val_aucs)
    fold_test_probs = np.array(fold_test_probs)
    print(f'\nOfficial fold val AUCs: {fold_val_aucs.round(4).tolist()}  '
          f'mean={fold_val_aucs.mean():.4f} +/- {fold_val_aucs.std():.4f}')

    ensemble_prob = np.mean(fold_test_probs, axis=0)
    best_fold = int(np.argmax(fold_val_aucs))
    single_prob = fold_test_probs[best_fold]
    print(f'Best single fold by val AUC: fold {best_fold} (val AUC={fold_val_aucs[best_fold]:.4f})')

    rows = []
    for cpt in [0.5, 0.8]:
        rows.append({**evaluate(y_test, ensemble_prob, f'{model}-CV Stage-{stage} ({stage_label}) Ensemble', cpt), 'Split': 'Test'})
        rows.append({**evaluate(y_test, single_prob, f'{model}-CV Stage-{stage} ({stage_label}) BestFold', cpt), 'Split': 'Test'})

    df = pd.DataFrame(rows)[['Split'] + METRIC_COLS]
    out_csv = os.path.join(OUT_DIR, f'classical_cv_stage{stage}_{model.lower()}_results.csv')
    df.to_csv(out_csv, index=False)
    np.save(os.path.join(OUT_DIR, f'classical_cv_stage{stage}_{model.lower()}_ensemble_test_probs.npy'), ensemble_prob)

    print(f'\n== {model} Stage-{stage} ({stage_label}) final held-out-223 results ==')
    print(df.to_string(index=False))
    print(f'\nSaved -> {out_csv}')


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in MODELS or sys.argv[2] not in ('1', '2'):
        print(f'Usage: python {sys.argv[0]} <{"|".join(MODELS)}> <1|2>')
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]))
