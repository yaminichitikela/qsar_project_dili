"""Trains ONE corrected graph transformer on ONE published split unit.

One job per (assay, architecture, protocol, unit), matching the per-architecture
job convention already used across this project.

WHAT IS HELD FIXED
------------------
Everything except the architecture, so the retrained rows differ from the
published ones by exactly one thing:

    splits          loaded from the published data dirs, never regenerated
    features        the 27-dim atom / 6-dim bond graphs already on disk
    configs         the same three GT_CONFIGS entries
    optimiser       Adam at the config's lr, no weight decay
    loss            BCEWithLogitsLoss with pos_weight = n_inactive / n_active
    batch size      64
    schedule        max 100 epochs, early stop patience 20 on val ROC-AUC,
                    best-epoch weights restored
    selection       highest val ROC-AUC across the three configs
    test            scored once, after selection

The training and evaluation loops are transcribed from gt_train / gt_eval in
Summary_1_models/modeling/wahab_jobs/gt_model.py; only the model call signatures
differ, because the corrected Graphormer consumes the structural encodings.

STRUCTURAL ENCODINGS ARE COMPUTED IN-JOB
----------------------------------------
True Graphormer needs per-atom-pair shortest-path distances and edge-path
features. Caching them to disk would add ~10 GB across 82 units, so they are
computed here instead: a few minutes of CPU per job, negligible beside three
configs of GPU training, and it keeps the published data dirs untouched.

GraphGPS reads the published graphs_*_gps.pt, which already carry the Laplacian
PE; the correction there is the random sign flipping applied during training
inside GraphGPSModel, not a change to the stored encoding.

Usage:
    python train_gt_corrected.py <mmp|hepg2|chok1> <Graphormer|GraphGPS> \
        <random_repeat|loo|scaffold|up_single|up_loo> <unit>

Outputs (./results/<assay>/<protocol>/):
    probs/<arch>_<unit>_test_probs.npy    -- consumed by compute_consensus_corrected.py
    <arch>_<unit>_metrics.csv
    <arch>_<unit>_config.json             -- winning config + per-config val AUCs
    <arch>_<unit>.pt
"""
import copy
import json
import os
import sys
import time
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data_map import ASSAY_LABEL, ASSAY_PROTOCOLS, PROTOCOLS, data_dir, out_dir, units
from eval_utils import METRIC_COLS, evaluate
from gt_model_corrected import (GT_CONFIGS, MAX_EPOCHS, PATIENCE, ATOM_DIM,
                                GraphGPSModel, TrueGraphormer,
                                precompute_structural_encodings)

SEED = 42
BATCH = 64
ARCHS = ('Graphormer', 'GraphGPS')


def load_graphs(assay, protocol, unit, arch):
    d = data_dir(assay, protocol, unit)
    suffix = '_gps' if arch == 'GraphGPS' else ''
    out = {}
    for split in ('train', 'val', 'test'):
        p = os.path.join(d, f'graphs_{split}{suffix}.pt')
        out[split] = torch.load(p, weights_only=False)
    return out


def add_structural_encodings(graphs, label):
    """Attaches spd_flat / edgepath_flat, reusing the result for graphs that are
    structurally identical (up-sampled splits repeat the same molecules many
    times, so the cache does most of the work there)."""
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


def build(arch, cfg, device):
    if arch == 'Graphormer':
        return TrueGraphormer(ATOM_DIM, hidden=cfg['hidden'],
                              num_heads=cfg['num_heads'],
                              num_layers=cfg['num_layers'],
                              dropout=cfg['dropout']).to(device)
    return GraphGPSModel(ATOM_DIM, hidden=cfg['hidden'],
                         num_heads=cfg['num_heads'],
                         num_layers=cfg['num_layers'],
                         dropout=cfg['dropout']).to(device)


def forward(model, arch, batch):
    if arch == 'Graphormer':
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                     batch.spd_flat, batch.edgepath_flat)
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe)


def train_epoch(model, arch, loader, opt, criterion, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        loss = criterion(forward(model, arch, batch), batch.y)
        loss.backward()
        opt.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


def evaluate_loader(model, arch, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits.append(forward(model, arch, batch).cpu())
            labels.append(batch.y.cpu())
    probs = torch.sigmoid(torch.cat(logits)).numpy()
    y = torch.cat(labels).numpy().astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float('nan')
    return probs, y, auc


def main(assay, arch, protocol, unit):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'=== {ASSAY_LABEL[assay]} | {arch} (CORRECTED) | {protocol} unit {unit}')
    print(f'    {PROTOCOLS[protocol]["caption"]}')
    print(f'    device={device}  data={data_dir(assay, protocol, unit)}', flush=True)

    graphs = load_graphs(assay, protocol, unit, arch)
    if arch == 'Graphormer':
        graphs = {k: add_structural_encodings(v, k) for k, v in graphs.items()}

    loaders = {k: DataLoader(v, batch_size=BATCH, shuffle=(k == 'train'))
               for k, v in graphs.items()}
    y_tr = np.array([g.y.item() for g in graphs['train']])
    n_pos = max(int((y_tr == 1).sum()), 1)
    pos_weight = torch.tensor([float((y_tr == 0).sum()) / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f'    train={len(y_tr)} val={len(graphs["val"])} test={len(graphs["test"])}'
          f'  pos_weight={pos_weight.item():.3f}', flush=True)

    best_auc, best_state, best_cfg, trial_log = -1.0, None, None, []
    for cfg in GT_CONFIGS[arch]:
        t0 = time.time()
        model = build(arch, cfg, device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        cfg_best, cfg_state, stale = -1.0, None, 0
        for epoch in range(1, MAX_EPOCHS + 1):
            train_epoch(model, arch, loaders['train'], opt, criterion, device)
            _, _, va = evaluate_loader(model, arch, loaders['val'], device)
            if va > cfg_best:
                cfg_best, cfg_state, stale = va, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        trial_log.append({**cfg, 'val_auc': round(float(cfg_best), 4),
                          'epochs_run': epoch, 'seconds': round(time.time() - t0, 1)})
        print(f'    {cfg}  val AUC={cfg_best:.4f}  stopped@ep{epoch}  '
              f'{time.time() - t0:.0f}s', flush=True)
        if cfg_best > best_auc:
            best_auc, best_state, best_cfg = cfg_best, cfg_state, cfg.copy()

    print(f'>>  best val AUC={best_auc:.4f}  cfg={best_cfg}', flush=True)

    model = build(arch, best_cfg, device)
    model.load_state_dict(best_state)
    probs_test, y_test, auc_test = evaluate_loader(model, arch, loaders['test'], device)
    probs_val, y_val, _ = evaluate_loader(model, arch, loaders['val'], device)
    print(f'    test ROC-AUC={auc_test:.4f}', flush=True)

    d = out_dir(assay, protocol)
    os.makedirs(os.path.join(d, 'probs'), exist_ok=True)
    np.save(os.path.join(d, 'probs', f'{arch}_{unit}_test_probs.npy'),
            probs_test.astype(np.float32))
    torch.save({'state_dict': best_state, 'config': best_cfg},
               os.path.join(d, f'{arch}_{unit}.pt'))
    with open(os.path.join(d, f'{arch}_{unit}_config.json'), 'w') as fh:
        json.dump({'winning': best_cfg, 'val_auc': best_auc, 'trials': trial_log}, fh, indent=2)

    rows = []
    for cpt in (0.5, 0.8):
        rows.append({**evaluate(y_val, probs_val, arch, cpt), 'Split': 'Val', 'Unit': unit})
        rows.append({**evaluate(y_test, probs_test, arch, cpt), 'Split': 'Test', 'Unit': unit})
    df = pd.DataFrame(rows)[['Unit', 'Split'] + METRIC_COLS]
    df.to_csv(os.path.join(d, f'{arch}_{unit}_metrics.csv'), index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] not in ASSAY_PROTOCOLS \
            or sys.argv[2] not in ARCHS or sys.argv[3] not in PROTOCOLS:
        print(f'Usage: python {os.path.basename(__file__)} '
              f'<mmp|hepg2|chok1> <{"|".join(ARCHS)}> <{"|".join(PROTOCOLS)}> <unit>')
        sys.exit(1)
    a, arch, proto, u = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    if proto not in ASSAY_PROTOCOLS[a]:
        print(f'{a} does not report protocol {proto} '
              f'(reports: {ASSAY_PROTOCOLS[a]})')
        sys.exit(1)
    if u not in units(a, proto):
        print(f'unit must be one of {units(a, proto)}')
        sys.exit(1)
    main(a, arch, proto, u)
