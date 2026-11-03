"""Smoke-tests trainer.run() for all ten families, with and without the in vitro
feature block, on a small slice of the real prepped data.

This is a wiring test (2 epochs, tiny subsets), not a performance test. It checks
that every family trains, that probabilities are well formed and the right
length, that out-of-split prediction targets work (the assay models must score
the DILI compounds), and that the in vitro block actually reaches the model.

Each family runs in its own subprocess, so a hard crash in one is reported rather
than killing the suite. That matters on macOS: importing torch before xgboost
segfaults there (duplicate OpenMP runtimes), which is why trainer.py imports
torch lazily and why the classical families are tested in a torch-free process.
Linux/Wahab is unaffected either way.

Usage: python test_trainer.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSICAL = ['RF', 'SVM', 'XGB']
GRAPH = ['GCN', 'GAT', 'GIN', 'SAGE', 'Graphormer', 'GraphGPS', 'MVGNN']

PARAMS = {
    'RF': dict(n_estimators=50, max_features='sqrt', min_samples_leaf=1),
    'SVM': dict(C=1.0, gamma='scale'),
    'XGB': dict(n_estimators=40, max_depth=3, learning_rate=0.1),
    'GCN': dict(hidden=32, n_layers=2, dropout=0.2, lr=1e-3),
    'GAT': dict(hidden=32, n_layers=2, dropout=0.2, lr=1e-3, heads=2),
    'GIN': dict(hidden=32, n_layers=2, dropout=0.2, lr=1e-3),
    'SAGE': dict(hidden=32, n_layers=2, dropout=0.2, lr=1e-3),
    'Graphormer': dict(hidden=32, num_heads=4, num_layers=2, dropout=0.2, lr=1e-3),
    'GraphGPS': dict(hidden=32, num_heads=4, num_layers=2, dropout=0.2, lr=1e-3),
    'MVGNN': dict(hidden=32, n_layers=2, dropout=0.2, lr=1e-3, lam=1.0),
}

WORKER = r'''
import json, os, sys
import numpy as np
sys.path.insert(0, HERE_PLACEHOLDER)
os.chdir(HERE_PLACEHOLDER)
import trainer

fam, params, use_extra = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3] == '1'
X = np.load('data_mmp/X_all.npy'); y = np.load('data_mmp/y_all.npy')
need_graphs = fam not in ('RF', 'SVM', 'XGB')
graphs = bonds = None
if need_graphs:
    import torch
    graphs = torch.load('data_mmp/graphs_all.pt', weights_only=False)
    bonds = torch.load('data_mmp/graphs_all_bond.pt', weights_only=False)

rng = np.random.RandomState(0)
act, ina = np.where(y == 1)[0], np.where(y == 0)[0]
tr = np.concatenate([rng.choice(act, 80, False), rng.choice(ina, 80, False)])
ra, ri = np.setdiff1d(act, tr), np.setdiff1d(ina, tr)
va = np.concatenate([rng.choice(ra, 30, False), rng.choice(ri, 30, False)])
ra, ri = np.setdiff1d(ra, va), np.setdiff1d(ri, va)
te = np.concatenate([rng.choice(ra, 30, False), rng.choice(ri, 30, False)])
idx = {'train': tr, 'val': va, 'test': te}
extra = rng.rand(len(y), 2).astype(np.float32) if use_extra else None

data = trainer.SplitData(X, y, graphs, bonds, idx, extra=extra)
if not need_graphs:
    ep = {'dili': (X[te[:10]], None if extra is None else extra[te[:10]])}
else:
    d2 = trainer.SplitData(X, y, graphs, bonds, {'t': te[:10]}, extra=extra)
    ep = ({'dili': (d2.graph_list('t'), d2.graph_list('t', bond=True))}
          if fam == 'MVGNN' else {'dili': d2.graph_list('t')})

auc, probs, model = trainer.run(fam, params, data, predict_sets=('val', 'test'),
                                extra_predict=ep, device='cpu',
                                max_epochs=2, patience=2)
ok = all(len(probs[k]) == n and np.all((probs[k] >= 0) & (probs[k] <= 1))
         and not np.isnan(probs[k]).any()
         for k, n in (('val', len(va)), ('test', len(te)), ('dili', 10)))
if need_graphs:
    head = model.head_a if fam == 'MVGNN' else model.head
    width = head[0].in_features
else:
    width = data.fp('train').shape[1]
print('RESULT' + json.dumps({'auc': auc, 'probs_ok': bool(ok), 'width': int(width)}))
'''


def run_worker(fam, use_extra):
    src = WORKER.replace('HERE_PLACEHOLDER', repr(HERE))
    env = dict(os.environ, PYTHONWARNINGS='ignore')
    p = subprocess.run([sys.executable, '-c', src, fam, json.dumps(PARAMS[fam]),
                        '1' if use_extra else '0'],
                       capture_output=True, text=True, env=env, cwd=HERE)
    for line in p.stdout.splitlines():
        if line.startswith('RESULT'):
            return json.loads(line[len('RESULT'):])
    return {'error': f'exit {p.returncode}: ' + (p.stderr.strip().splitlines() or ['no output'])[-1]}


if not os.path.exists(f'{HERE}/data_mmp/X_all.npy'):
    print('run prepare_assay_noDILI.py mmp first')
    sys.exit(1)

fails = []
print(f'{"family":12s} {"val AUC":>9s} {"val AUC +iv":>12s}  {"probs":>6s}  '
      f'{"input width":>22s}  in vitro consumed')
for fam in CLASSICAL + GRAPH:
    plain, extra = run_worker(fam, False), run_worker(fam, True)
    if 'error' in plain or 'error' in extra:
        msg = plain.get('error') or extra.get('error')
        print(f'{fam:12s} ERROR: {msg}')
        fails.append(f'{fam}: {msg}')
        continue
    consumed = (extra['width'] - plain['width']) == 2
    probs_ok = plain['probs_ok'] and extra['probs_ok']
    print(f'{fam:12s} {plain["auc"]:9.4f} {extra["auc"]:12.4f}  '
          f'{"ok" if probs_ok else "BAD":>6s}  {plain["width"]:>10d} -> {extra["width"]:<9d}  '
          f'{"yes" if consumed else "NO"}')
    if not probs_ok:
        fails.append(f'{fam}: malformed probabilities')
    if not consumed:
        fails.append(f'{fam}: in vitro features not consumed')

print('\n' + '=' * 78)
print('ALL 10 FAMILIES OK' if not fails else 'FAILURES:\n  - ' + '\n  - '.join(fails))
sys.exit(1 if fails else 0)
