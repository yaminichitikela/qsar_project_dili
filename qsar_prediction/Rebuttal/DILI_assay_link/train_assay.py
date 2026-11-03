"""STEP B -- trains ONE assay model family and predicts the DILI compounds.

One family per job, matching the convention already used in
DILI_proxy_clean/modeling/wahab_jobs/jobs/ (one slurm file per architecture) and
in the summary-assay folders. Classical families run on CPU, graph families on
GPU, and a failure in one family leaves the others untouched.

TWO MODES
---------
  search          Optuna hyperparameter search, ONCE per (assay, family), on
                  repeat 0's train/val split. Writes the winning config.
  fit <repeat>    Trains that repeat with the winning config and predicts the
                  repeat's val/test plus every DILI compound.

Searching once rather than inside all 21 repeats is the principle this project
already follows -- DILI_mvgnn_cv/train_classical_cv.py calls it "one
hyperparameter decision, not per-fold decisions". It also cuts the work from
~13,650 model fits to ~1,070, and removes hyperparameter-selection noise between
repeats, which would otherwise be confounded with the OOB averaging.

Optuna uses the TPE sampler with median pruning, reporting validation ROC-AUC
each epoch through trainer's epoch_callback. Pruning matters here because true
Graphormer's dense attention is costlier per epoch than the old 1-hop version.

WHY EVERY REPEAT PREDICTS EVERY DILI COMPOUND
---------------------------------------------
build_invitro_features.py decides per compound which repeats may contribute: for
a compound that is itself in the assay data, only repeats where it was held out
(the OOB mask); for a compound absent from the assay data, all repeats, since no
model ever saw it. Both cases need the full prediction matrix, so each repeat
scores all 13,241 DILI compounds and the aggregation happens later.

Usage:
    python train_assay.py <mmp|caspase> <family> search [n_trials]
    python train_assay.py <mmp|caspase> <family> fit <repeat>

Families: RF SVM XGB GCN GAT GIN SAGE Graphormer GraphGPS MVGNN

Outputs (./results/<assay>/):
    <family>_winning_config.json          -- search mode
    <family>_optuna_trials.csv            -- search mode
    probs/<family>_r<r>_dili.npy          -- fit mode, [13241] DILI predictions
    probs/<family>_r<r>_test.npy          -- fit mode, that repeat's test probs
    <family>_r<r>_metrics.csv             -- fit mode, val/test metrics
"""
import json
import os
import sys
import time
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import trainer
from eval_utils import METRIC_COLS, evaluate
from optuna_spaces import ALL_FAMILIES, CLASSICAL_FAMILIES, n_trials as default_trials, suggest

SEED = 42
MAX_EPOCHS = 100
PATIENCE = 20
SEARCH_REPEAT = 0        # the split the hyperparameter search is done on


def load_assay(assay, need_graphs):
    d = os.path.join(HERE, f'data_{assay}')
    X = np.load(f'{d}/X_all.npy')
    y = np.load(f'{d}/y_all.npy')
    graphs = bonds = None
    if need_graphs:
        import torch
        graphs = torch.load(f'{d}/graphs_all.pt', weights_only=False)
        bonds = torch.load(f'{d}/graphs_all_bond.pt', weights_only=False)
    return X, y, graphs, bonds


def load_repeat_idx(assay, r):
    d = os.path.join(HERE, f'data_{assay}', f'repeat_{r}')
    return {s: np.load(f'{d}/idx_{s}.npy') for s in ('train', 'val', 'test')}


def n_repeats(assay):
    d = os.path.join(HERE, f'data_{assay}')
    return len([x for x in os.listdir(d) if x.startswith('repeat_')])


def dili_targets(family):
    """The DILI prediction targets, in the form this family consumes."""
    d = os.path.join(HERE, 'data_dili_features')
    if family in CLASSICAL_FAMILIES:
        return (np.load(f'{d}/X_dili.npy'), None)
    import torch
    graphs = torch.load(f'{d}/graphs_dili.pt', weights_only=False)
    if family == 'MVGNN':
        bonds = torch.load(f'{d}/graphs_dili_bond.pt', weights_only=False)
        return (graphs, bonds)
    return graphs


def do_search(assay, family, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    need_graphs = family not in CLASSICAL_FAMILIES
    X, y, graphs, bonds = load_assay(assay, need_graphs)
    idx = load_repeat_idx(assay, SEARCH_REPEAT)
    data = trainer.SplitData(X, y, graphs, bonds, idx)
    device = 'cuda' if need_graphs and _cuda() else 'cpu'
    print(f'{assay}/{family} search: {n_trials} trials on repeat {SEARCH_REPEAT}, '
          f'device={device}', flush=True)

    log = []

    def objective(trial):
        params = suggest(trial, family)
        t0 = time.time()

        def cb(epoch, best_auc):
            trial.report(best_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        auc, _, _ = trainer.run(family, params, data, predict_sets=('val',),
                                device=device, max_epochs=MAX_EPOCHS,
                                patience=PATIENCE,
                                epoch_callback=cb if need_graphs else None,
                                calibrate=False)
        log.append({**params, 'val_auc': auc, 'seconds': round(time.time() - t0, 1),
                    'trial': trial.number})
        print(f'  trial {trial.number:3d}  val AUC={auc:.4f}  '
              f'{time.time() - t0:.0f}s  {params}', flush=True)
        return auc

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))

    out = os.path.join(HERE, 'results', assay)
    os.makedirs(out, exist_ok=True)
    best = dict(study.best_trial.params)
    # the stored config must be what the model constructor expects, so expand the
    # paired hidden_heads choice the search space uses for the transformers
    if 'hidden_heads' in best:
        hidden, heads = (int(v) for v in best.pop('hidden_heads').split('_'))
        best['hidden'], best['num_heads'] = hidden, heads
    with open(f'{out}/{family}_winning_config.json', 'w') as fh:
        json.dump(best, fh, indent=2)
    pd.DataFrame(log).to_csv(f'{out}/{family}_optuna_trials.csv', index=False)
    print(f'\nBest val AUC={study.best_value:.4f}\nconfig={best}\n-> {out}', flush=True)


def do_fit(assay, family, repeat):
    need_graphs = family not in CLASSICAL_FAMILIES
    out = os.path.join(HERE, 'results', assay)
    cfg_path = f'{out}/{family}_winning_config.json'
    if not os.path.exists(cfg_path):
        print(f'missing {cfg_path} -- run search mode first')
        sys.exit(1)
    params = json.load(open(cfg_path))

    X, y, graphs, bonds = load_assay(assay, need_graphs)
    idx = load_repeat_idx(assay, repeat)
    data = trainer.SplitData(X, y, graphs, bonds, idx)
    device = 'cuda' if need_graphs and _cuda() else 'cpu'
    print(f'{assay}/{family} fit repeat {repeat}: device={device} config={params}',
          flush=True)

    t0 = time.time()
    val_auc, probs, _ = trainer.run(
        family, params, data, predict_sets=('val', 'test'),
        extra_predict={'dili': dili_targets(family)},
        device=device, max_epochs=MAX_EPOCHS, patience=PATIENCE)
    print(f'  val AUC={val_auc:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    probs_dir = f'{out}/probs'
    os.makedirs(probs_dir, exist_ok=True)
    np.save(f'{probs_dir}/{family}_r{repeat}_dili.npy', probs['dili'].astype(np.float32))
    np.save(f'{probs_dir}/{family}_r{repeat}_test.npy', probs['test'].astype(np.float32))

    rows = []
    for cpt in (0.5, 0.8):
        for split in ('val', 'test'):
            rows.append({**evaluate(data.labels(split), probs[split], family, cpt),
                         'Split': split, 'Repeat': repeat})
    df = pd.DataFrame(rows)[['Repeat', 'Split'] + METRIC_COLS]
    df.to_csv(f'{out}/{family}_r{repeat}_metrics.csv', index=False)
    print(df.to_string(index=False), flush=True)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def usage():
    print(f'Usage: python {os.path.basename(__file__)} <mmp|caspase> '
          f'<{"|".join(ALL_FAMILIES)}> search [n_trials]')
    print(f'       python {os.path.basename(__file__)} <mmp|caspase> <family> fit <repeat>')
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 4 or sys.argv[1] not in ('mmp', 'caspase') \
            or sys.argv[2] not in ALL_FAMILIES:
        usage()
    assay, family, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    if mode == 'search':
        nt = int(sys.argv[4]) if len(sys.argv) > 4 else default_trials(family)
        do_search(assay, family, nt)
    elif mode == 'fit':
        if len(sys.argv) < 5:
            usage()
        r = int(sys.argv[4])
        if not 0 <= r < n_repeats(assay):
            print(f'repeat must be 0..{n_repeats(assay) - 1}')
            sys.exit(1)
        do_fit(assay, family, r)
    else:
        usage()
