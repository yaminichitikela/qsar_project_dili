"""Corrected graph transformers on Caspase-3/7 CHO-K1 (AID 1347037), 1:5 downsampled training set.
Usage: python train_gt.py <scaffold> <Graphormer|GraphGPS>
Requires prepare_scaffold_1to5.py (and prepare_gt_pe_protocol.py for
GraphGPS) to have been run first.

Uses the CORRECTED architectures in gt_model_corrected.py (copied from
../../graph_transformer_corrected/): Graphormer with global all-pairs attention,
centrality, shortest-path spatial and edge-path encodings and a [VNode]
readout; GraphGPS with random Laplacian-PE sign flipping during training.
Training loop as in ../../graph_transformer_corrected/train_gt_corrected.py.
"""
import copy
import os
import sys
import time
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from eval_utils import evaluate, METRIC_COLS
from protocols import from_argv, unit_dir, prob_name

P = from_argv('<Graphormer|GraphGPS>')
from gt_model_corrected import (GT_CONFIGS, MAX_EPOCHS, PATIENCE, ATOM_DIM,
                                GraphGPSModel, TrueGraphormer,
                                precompute_structural_encodings)
from sklearn.metrics import roc_auc_score

SEED = 42
BATCH = 64

if len(sys.argv) != 2 or sys.argv[1] not in GT_CONFIGS:
    print(f'Usage: python {sys.argv[0]} <scaffold> <Graphormer|GraphGPS>')
    sys.exit(1)
ARCH = sys.argv[1]
SUFFIX = '_gps' if ARCH == 'GraphGPS' else ''

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Architecture: {ARCH} (corrected)  Device: {DEVICE}', flush=True)

OUT_DIR = P['out_dir']
os.makedirs(os.path.join(OUT_DIR, 'probs'), exist_ok=True)


def add_structural_encodings(graphs, label):
    """Attaches spd_flat / edgepath_flat for the corrected Graphormer, reusing
    the result for structurally identical graphs (the train set may
    repeat the same molecule)."""
    t0, cache, out = time.time(), {}, []
    for g in graphs:
        key = (g.x.shape[0], hash(g.edge_index.numpy().tobytes()),
               hash(g.edge_attr.numpy().tobytes()) if g.edge_attr is not None else 0)
        if key not in cache:
            cache[key] = precompute_structural_encodings(g)
        spd, ep = cache[key]
        out.append(Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr,
                        y=g.y, spd_flat=spd.to(torch.int16),
                        edgepath_flat=ep.to(torch.float16)))
    print(f'  encodings {label}: {len(out)} graphs, {len(cache)} unique '
          f'({time.time() - t0:.0f}s)', flush=True)
    return out


def build(cfg):
    cls = TrueGraphormer if ARCH == 'Graphormer' else GraphGPSModel
    return cls(ATOM_DIM, hidden=cfg['hidden'], num_heads=cfg['num_heads'],
               num_layers=cfg['num_layers'], dropout=cfg['dropout']).to(DEVICE)


def forward(model, batch):
    if ARCH == 'Graphormer':
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                     batch.spd_flat, batch.edgepath_flat)
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe)


def train_epoch(model, loader, opt, criterion):
    model.train()
    for batch in loader:
        batch = batch.to(DEVICE)
        opt.zero_grad()
        loss = criterion(forward(model, batch), batch.y)
        loss.backward()
        opt.step()


def evaluate_loader(model, loader):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits.append(forward(model, batch).cpu())
            labels.append(batch.y.cpu())
    probs = torch.sigmoid(torch.cat(logits)).numpy()
    y = torch.cat(labels).numpy().astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float('nan')
    return probs, y, auc


repeat_rows = []
per_repeat_test_metrics = {0.5: [], 0.8: []}

for rep in range(P['n']):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rep_dir = unit_dir(P, rep)
    graphs = {s: torch.load(os.path.join(rep_dir, f'graphs_{s}{SUFFIX}.pt'), weights_only=False)
              for s in ('train', 'val', 'test')}
    if ARCH == 'Graphormer':
        graphs = {s: add_structural_encodings(g, f'{P["unit"]} {rep} {s}') for s, g in graphs.items()}

    loaders = {s: DataLoader(g, batch_size=BATCH, shuffle=(s == 'train')) for s, g in graphs.items()}
    y_tr = np.array([g.y.item() for g in graphs['train']])
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                              dtype=torch.float, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc, best_state, best_cfg = -1.0, None, None
    for cfg in GT_CONFIGS[ARCH]:
        t0 = time.time()
        model = build(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        cfg_best, cfg_state, stale = -1.0, None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(model, loaders['train'], opt, criterion)
            _, _, va = evaluate_loader(model, loaders['val'])
            if va > cfg_best:
                cfg_best, cfg_state, stale = va, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        print(f'  {P["unit"]} {rep} {cfg}  val AUC={cfg_best:.4f}  stopped@ep{epoch}  '
              f'{time.time() - t0:.0f}s', flush=True)
        if cfg_best > best_auc:
            best_auc, best_state, best_cfg = cfg_best, cfg_state, cfg.copy()

    print(f'{P["unit_col"]} {rep}: best val AUC={best_auc:.4f}  cfg={best_cfg}', flush=True)

    model = build(best_cfg)
    model.load_state_dict(best_state)
    prob_val, y_val, _ = evaluate_loader(model, loaders['val'])
    prob_test, y_test, _ = evaluate_loader(model, loaders['test'])

    # save this unit's winning test-set probs, needed by compute_consensus.py
    np.save(os.path.join(OUT_DIR, 'probs', prob_name(P, ARCH.lower(), rep)), prob_test)

    for cpt in [0.5, 0.8]:
        row_val = {**evaluate(y_val, prob_val, ARCH, cpt), 'Split': 'Val', P['unit_col']: rep}
        row_test = {**evaluate(y_test, prob_test, ARCH, cpt), 'Split': 'Test', P['unit_col']: rep}
        repeat_rows.append(row_val)
        repeat_rows.append(row_test)
        per_repeat_test_metrics[cpt].append(row_test)

df_reps = pd.DataFrame(repeat_rows)[[P['unit_col'], 'Split'] + METRIC_COLS]
df_reps.to_csv(os.path.join(OUT_DIR, f'{ARCH.lower()}_per_unit.csv'), index=False)

summary_rows = []
numeric_cols = ['Sensitivity', 'Specificity', 'CCR', 'ROC-AUC', 'PR-AUC', 'MCC', 'Bal-Acc', 'F1']
for cpt in [0.5, 0.8]:
    test_df = pd.DataFrame(per_repeat_test_metrics[cpt])
    summary_rows.append({'Model': ARCH, 'CPT': cpt, 'Stat': 'mean', **test_df[numeric_cols].mean().round(4).to_dict()})
    summary_rows.append({'Model': ARCH, 'CPT': cpt, 'Stat': 'std', **test_df[numeric_cols].std().round(4).to_dict()})

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, f'{ARCH.lower()}_summary.csv'), index=False)
print(f'\n== Mean +/- std across {P["n"]} {P["unit"]}s (Test set) ==')
print(df_summary)
print(f'\nSaved -> {OUT_DIR}/{ARCH.lower()}_per_unit.csv and {ARCH.lower()}_summary.csv')
