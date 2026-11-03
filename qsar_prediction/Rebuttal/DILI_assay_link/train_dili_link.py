"""STEP D -- the linking experiment itself: does knowing the in vitro HepG2
mechanisms improve DILI prediction?

THE COMPARISON
--------------
Two arms, identical in every respect except their inputs:

    structure      ECFP4 fingerprints / molecular graphs only
    structure_iv   the same, plus two molecule-level scalars:
                   P(MMP active) and P(caspase-3/7 active)

Both arms use the same splits, the same hyperparameter search protocol, the same
early-stopping rule, the same metric, and the same fixed Held-223 test set. That
single difference in inputs is the experiment, so any gap between the arms is
attributable to the in vitro features and nothing else.

Two stages, mirroring the published DILI pipeline:

    stage 1   "Proxy-only"  -- train on the clinical pool (Alloriginal minus the
                               gold sets); this is what the published Stage-1
                               tables report
    stage 2   "Proxy+DILI"  -- add NCV-888. Graph families warm-start from their
                               stage-1 weights (as train_gnn_stage2.py does);
                               classical families re-fit independently, having no
                               continuable weights

"Proxy" in those stage names refers to the weak clinical DILI labels, NOT to the
in vitro assays -- the two in vitro columns are always called "in vitro" here so
the published stage names keep their meaning.

WHY THE IN VITRO VALUES ARE SAFE TO USE
---------------------------------------
Each value came from assay models trained without that compound (out-of-bag over
21 repeated assay splits; see build_invitro_features.py and audit_leakage.py).
No DILI label ever entered an assay model, and no assay model ever saw the
compound it is scoring. Without that, 76% of MMP and 69% of caspase overlap the
DILI universe and the features would be partly memorised assay labels.

Usage:
    python train_dili_link.py <structure|structure_iv> <family> <stage> search
    python train_dili_link.py <structure|structure_iv> <family> <stage> fit <repeat>

Outputs (./results/link/<arm>/):
    <family>_s<stage>_winning_config.json
    <family>_s<stage>_optuna_trials.csv
    probs/<family>_s<stage>_r<r>_test.npy     -- Held-223 probabilities
    states/<family>_s1_r<r>.pt                -- stage-1 weights, for the warm start
    <family>_s<stage>_r<r>_metrics.csv
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
DATA = os.path.join(HERE, 'data_dili_link')
ARMS = ('structure', 'structure_iv')
# Stage 1 trains from scratch; stage 2 is a shorter fine-tune, matching the
# schedules in DILI_proxy_clean (stage 1: 80/15, stage 2: 60/12).
EPOCHS = {1: (80, 15), 2: (60, 12)}
SEARCH_REPEAT = 0


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def n_repeats():
    return len([x for x in os.listdir(DATA) if x.startswith('repeat_')])


def build_split_data(arm, family, stage, repeat):
    """Concatenates clin / ncv / test into one block and indexes the splits into it.

    SplitData addresses every split as indices into shared arrays, so the three
    DILI parts are stacked once (clin, then ncv, then test) and each split is
    expressed as offsets. Stage 2 simply adds the whole NCV block to the training
    indices; validation stays the clinical val split and the test set is always
    Held-223, untouched by either stage.
    """
    need_graphs = family not in CLASSICAL_FAMILIES
    parts, off, cursor = {}, {}, 0
    Xs, ys, ivs, graphs, bonds = [], [], [], [], []
    for name in ('clin', 'ncv', 'test'):
        X = np.load(f'{DATA}/X_{name}.npy')
        y = np.load(f'{DATA}/y_{name}.npy')
        iv = np.load(f'{DATA}/iv_{name}.npy')
        off[name] = cursor
        cursor += len(y)
        parts[name] = len(y)
        Xs.append(X); ys.append(y); ivs.append(iv)
        if need_graphs:
            import torch
            graphs += torch.load(f'{DATA}/graphs_{name}.pt', weights_only=False)
            bonds += torch.load(f'{DATA}/graphs_{name}_bond.pt', weights_only=False)

    X_all = np.vstack(Xs)
    y_all = np.concatenate(ys)
    iv_all = np.vstack(ivs).astype(np.float32)

    tr = off['clin'] + np.load(f'{DATA}/repeat_{repeat}/idx_train.npy')
    va = off['clin'] + np.load(f'{DATA}/repeat_{repeat}/idx_val.npy')
    te = off['test'] + np.arange(parts['test'])
    if stage == 2:
        tr = np.concatenate([tr, off['ncv'] + np.arange(parts['ncv'])])

    extra = iv_all if arm == 'structure_iv' else None
    data = trainer.SplitData(X_all, y_all, graphs or None, bonds or None,
                             {'train': tr, 'val': va, 'test': te}, extra=extra)
    return data, parts


def out_dir(arm):
    d = os.path.join(HERE, 'results', 'link', arm)
    os.makedirs(os.path.join(d, 'probs'), exist_ok=True)
    os.makedirs(os.path.join(d, 'states'), exist_ok=True)
    return d


def do_search(arm, family, stage, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    need_graphs = family not in CLASSICAL_FAMILIES
    data, parts = build_split_data(arm, family, stage, SEARCH_REPEAT)
    device = 'cuda' if need_graphs and _cuda() else 'cpu'
    max_epochs, patience = EPOCHS[stage]
    print(f'{arm}/{family} stage {stage} search: {n_trials} trials, device={device}')
    print(f'  train={len(data.idx["train"])} val={len(data.idx["val"])} '
          f'test={len(data.idx["test"])} extra_dim={data.extra_dim}', flush=True)

    log = []

    def objective(trial):
        params = suggest(trial, family)
        t0 = time.time()

        def cb(epoch, best):
            trial.report(best, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        auc, _, _ = trainer.run(family, params, data, predict_sets=('val',),
                                device=device, max_epochs=max_epochs,
                                patience=patience,
                                epoch_callback=cb if need_graphs else None,
                                calibrate=False)
        log.append({**params, 'val_auc': auc, 'seconds': round(time.time() - t0, 1),
                    'trial': trial.number})
        print(f'  trial {trial.number:3d}  val AUC={auc:.4f}  '
              f'{time.time() - t0:.0f}s', flush=True)
        return auc

    study = optuna.create_study(
        direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))
    study.optimize(objective, n_trials=n_trials, catch=(RuntimeError,))

    d = out_dir(arm)
    best = dict(study.best_trial.params)
    if 'hidden_heads' in best:
        hidden, heads = (int(v) for v in best.pop('hidden_heads').split('_'))
        best['hidden'], best['num_heads'] = hidden, heads
    with open(f'{d}/{family}_s{stage}_winning_config.json', 'w') as fh:
        json.dump(best, fh, indent=2)
    pd.DataFrame(log).to_csv(f'{d}/{family}_s{stage}_optuna_trials.csv', index=False)
    print(f'\nBest val AUC={study.best_value:.4f}\nconfig={best}', flush=True)


def do_fit(arm, family, stage, repeat):
    need_graphs = family not in CLASSICAL_FAMILIES
    d = out_dir(arm)
    cfg_path = f'{d}/{family}_s{stage}_winning_config.json'
    if not os.path.exists(cfg_path) and stage == 2 and need_graphs:
        # graph families fine-tune stage 1 rather than re-searching, so the
        # stage-1 config is the right one to continue with
        fallback = f'{d}/{family}_s1_winning_config.json'
        if os.path.exists(fallback):
            print(f'  stage 2: inheriting the stage-1 config ({fallback})')
            cfg_path = fallback
    if not os.path.exists(cfg_path):
        print(f'missing {cfg_path} -- run search first')
        sys.exit(1)
    params = json.load(open(cfg_path))
    data, parts = build_split_data(arm, family, stage, repeat)
    device = 'cuda' if need_graphs and _cuda() else 'cpu'
    max_epochs, patience = EPOCHS[stage]

    # stage 2 warm start from this arm/family/repeat's stage-1 weights
    init_state = None
    if stage == 2 and need_graphs:
        sp = f'{d}/states/{family}_s1_r{repeat}.pt'
        if os.path.exists(sp):
            import torch
            init_state = torch.load(sp, map_location=device, weights_only=False)
            print(f'  warm start from {sp}')
        else:
            print(f'  WARNING: {sp} missing, training stage 2 from scratch')

    print(f'{arm}/{family} stage {stage} repeat {repeat}: device={device} '
          f'train={len(data.idx["train"])} extra_dim={data.extra_dim}', flush=True)
    t0 = time.time()
    val_auc, probs, model = trainer.run(
        family, params, data, predict_sets=('val', 'test'), device=device,
        max_epochs=max_epochs, patience=patience, init_state=init_state)
    print(f'  val AUC={val_auc:.4f}  ({time.time() - t0:.0f}s)', flush=True)

    np.save(f'{d}/probs/{family}_s{stage}_r{repeat}_test.npy',
            probs['test'].astype(np.float32))
    if stage == 1 and need_graphs:
        import torch
        torch.save(model.state_dict(), f'{d}/states/{family}_s1_r{repeat}.pt')

    rows = []
    for cpt in (0.5, 0.8):
        for split in ('val', 'test'):
            rows.append({**evaluate(data.labels(split), probs[split], family, cpt),
                         'Split': split, 'Repeat': repeat, 'Arm': arm, 'Stage': stage})
    df = pd.DataFrame(rows)[['Arm', 'Stage', 'Repeat', 'Split'] + METRIC_COLS]
    df.to_csv(f'{d}/{family}_s{stage}_r{repeat}_metrics.csv', index=False)
    print(df.to_string(index=False), flush=True)


def usage():
    print(f'Usage: python {os.path.basename(__file__)} <{"|".join(ARMS)}> '
          f'<{"|".join(ALL_FAMILIES)}> <1|2> search [n_trials]')
    print(f'       python {os.path.basename(__file__)} <arm> <family> <1|2> fit <repeat>')
    sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 5 or sys.argv[1] not in ARMS or sys.argv[2] not in ALL_FAMILIES \
            or sys.argv[3] not in ('1', '2'):
        usage()
    arm, family, stage, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    if not os.path.exists(f'{DATA}/X_clin.npy'):
        print(f'missing {DATA} -- run prepare_dili_link_data.py first')
        sys.exit(1)
    if mode == 'search':
        nt = int(sys.argv[5]) if len(sys.argv) > 5 else default_trials(family)
        do_search(arm, family, stage, nt)
    elif mode == 'fit':
        if len(sys.argv) < 6:
            usage()
        r = int(sys.argv[5])
        if not 0 <= r < n_repeats():
            print(f'repeat must be 0..{n_repeats() - 1}')
            sys.exit(1)
        do_fit(arm, family, stage, r)
    else:
        usage()
