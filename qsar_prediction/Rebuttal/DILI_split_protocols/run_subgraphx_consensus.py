"""SubgraphX structural alerts for the Consensus-GNN ensemble on DILI.

Companion to run_subgraphx.py, which explains the two graph transformers
individually and then intersects their atom sets. That intersection is a
post-hoc reconciliation of two separate searches, and where the two models
share no atoms it has to fall back to the union. Here the ensemble is treated
as the model: the Shapley search runs once, against the mean of the four GNNs'
predicted probabilities, so the returned subgraph is the one the ensemble
itself relies on and no fallback is needed.

WHY CONSENSUS-GNN AND NOT THE NINE-MODEL CONSENSUS
--------------------------------------------------
SubgraphX scores candidate subgraphs by re-running the model on them, so it
applies directly to models that consume the molecular graph. The nine-model
consensus also contains RF, SVM and XGB, which consume ECFP4 fingerprints; those
could only be scored by recomputing a fingerprint for each masked substructure,
a stronger assumption than masking a graph. Consensus-GNN is the largest
ensemble that needs no such extension.

TARGET RUN
----------
10-fold CV with 1:1 downsampled training, fold 9, Stage 2 (Proxy+DILI) -- the
setting the manuscript reports for Consensus-GNN (ROC-AUC 0.6315 +/- 0.0241
over the ten folds). Fold 9's own Consensus-GNN ROC-AUC (0.6202) is the closest
of the ten to that mean, so the explained model is representative of the
reported number.

The ten folds are ten separately trained models; the manuscript's number is
their mean, not one model's score. Explaining one fold therefore explains one
draw from that distribution. The alternative -- the 10-fold ensemble
(ROC-AUC 0.6441), which averages all ten folds' probabilities -- would require
training and scoring forty models per candidate subgraph, which is why a single
representative fold is used instead.

Usage: python run_subgraphx_consensus.py [n_compounds|all] [n_rollouts]

Outputs (results/xai/):
    subgraphx_consensus_gnn.csv          one row per explained compound
    consensusgnn_tp<k>.svg               per-compound drawing
"""
import os
import random
import sys
import time
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CONFIGS, GNN, data_dir
from run_subgraphx import draw, largest_connected
from torch_geometric.data import Data
from torch_geometric.utils import subgraph as pyg_subgraph

SEED = 42
PROTO, UNIT, STAGE = 'cv', 9, 2
OUT = os.path.join(HERE, 'results', 'xai')
STATES = os.path.join(HERE, 'results', PROTO, 'states')


def build_models(device='cpu'):
    """The four fitted GNNs, in eval mode, as {family: model}."""
    import json
    from gnn_model import MolGNN, ATOM_DIM
    models = {}
    for family in GNN:
        p = f'{CONFIGS}/{family}_s{STAGE}_winning_config.json'
        if not os.path.exists(p):
            p = f'{CONFIGS}/{family}_s1_winning_config.json'   # as in load_config
        cfg = json.load(open(p))
        m = MolGNN(ATOM_DIM, hidden=cfg['hidden'], n_layers=cfg['n_layers'],
                   dropout=cfg['dropout'], conv_type=family,
                   heads=cfg.get('heads', 4))
        w = f'{STATES}/{family}_s{STAGE}_u{UNIT}.pt'
        if not os.path.exists(w):
            raise SystemExit(f'missing weights {w} -- run run_xai_gnn_models.sh first')
        m.load_state_dict(torch.load(w, map_location=device, weights_only=False))
        models[family] = m.to(device).eval()
    return models


@torch.no_grad()
def consensus_score(models, x, edge_index, nodes, device='cpu'):
    """Mean P(DILI) over the four GNNs for the induced subgraph on `nodes`."""
    if not nodes:
        return 0.0
    idx = torch.tensor(sorted(nodes), dtype=torch.long)
    ei, _ = pyg_subgraph(idx, edge_index, relabel_nodes=True, num_nodes=x.size(0))
    sub = Data(x=x[idx], edge_index=ei)
    batch = torch.zeros(sub.num_nodes, dtype=torch.long, device=device)
    ps = [torch.sigmoid(m(sub.x.to(device), sub.edge_index.to(device),
                          batch)).reshape(-1)[0]
          for m in models.values()]
    return float(torch.stack(ps).mean())


def explain(models, g, min_atoms=3, rollouts=20, device='cpu'):
    """Monte-Carlo Shapley over atom permutations; returns a 0/1 atom mask.

    Identical in substance to run_subgraphx.explain, but the value of a
    coalition is the ensemble mean rather than one model's output.
    """
    x, ei = g.x.cpu(), g.edge_index.cpu()
    n = x.size(0)
    if n <= min_atoms:
        return torch.ones(n)
    shap = torch.zeros(n)
    for _ in range(rollouts):
        perm = list(range(n)); random.shuffle(perm)
        coalition, prev = [], 0.0
        for node in perm:
            coalition.append(node)
            sc = consensus_score(models, x, ei, coalition, device)
            shap[node] += (sc - prev) / rollouts
            prev = sc
    order = shap.argsort(descending=True)
    k = max(min_atoms, int((shap > shap.median()).sum()))
    keep = largest_connected(order[:k].tolist(), ei)
    mask = torch.zeros(n); mask[keep] = 1.0
    return mask


@torch.no_grad()
def predict_all(models, graphs, device='cpu'):
    """Consensus probability for every whole molecule."""
    out = []
    for g in graphs:
        batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
        ps = [torch.sigmoid(m(g.x.to(device), g.edge_index.to(device),
                              batch)).reshape(-1)[0]
              for m in models.values()]
        out.append(float(torch.stack(ps).mean()))
    return np.asarray(out)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else '6'
    explain_all = (arg == 'all')
    n_top = 10 ** 9 if explain_all else int(arg)
    rollouts = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    d = data_dir()
    graphs = torch.load(f'{d}/graphs_test.pt', weights_only=False)
    y = np.load(f'{d}/y_test.npy').astype(int)
    keys = [ln.strip() for ln in open(f'{d}/ikeys_test.txt') if ln.strip()]
    lab = pd.read_csv(os.path.join(HERE, 'baseline', 'dili_labels.csv'))
    smi = lab.drop_duplicates('InChIKey').set_index('InChIKey').loc[keys, 'csmi'].tolist()

    models = build_models()
    p = predict_all(models, graphs)
    tp_all = np.where((p >= 0.5) & (y == 1))[0]
    tp = tp_all[np.argsort(-p[tp_all])][:n_top]
    print(f'Consensus-GNN: {len(tp_all)} true positives, '
          f'explaining the top {len(tp)}', flush=True)

    rows, t0 = [], time.time()
    for rank, i in enumerate(tp, 1):
        mask = explain(models, graphs[i], rollouts=rollouts)
        svg = f'{OUT}/consensusgnn_tp{rank}.svg'
        frag = draw(smi[i], mask, svg)
        rows.append(dict(model='Consensus-GNN', rank=rank, inchikey=keys[i],
                         smiles=smi[i], prob=round(float(p[i]), 4),
                         n_atoms=int(graphs[i].num_nodes),
                         n_highlighted=int(mask.sum()), fragment=frag,
                         svg=os.path.basename(svg),
                         mask=' '.join(str(int(v)) for v in mask.tolist())))
        if not explain_all:
            print(f'  {rank}. p={p[i]:.3f}  {int(mask.sum())}/{graphs[i].num_nodes}'
                  f' atoms  {frag}', flush=True)
        elif rank % 25 == 0:
            print(f'    ...{rank}/{len(tp)}', flush=True)
    print(f'  explained {len(tp)} compounds in {time.time() - t0:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/subgraphx_consensus_gnn.csv', index=False)
    print(f'\n-> {OUT}/subgraphx_consensus_gnn.csv')


if __name__ == '__main__':
    main()
