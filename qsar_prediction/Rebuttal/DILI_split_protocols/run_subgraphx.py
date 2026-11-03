"""SubgraphX structural alerts for the CORRECTED graph transformers on DILI.

Replaces the explanations in liver/xai.ipynb, which were produced with the
superseded architectures (TransformerConv-based "Graphormer-inspired" blocks
and a sign-arbitrary Laplacian PE). Those explanations describe models the
manuscript no longer reports, so they are regenerated here against the true
Graphormer (global all-pairs attention, spatial and edge encodings, [VNode]
readout) and the sign-flip-augmented GraphGPS, trained under the setting where the graph transformers lead: the up-sampled
repeated SCAFFOLD split, repeat 0, Stage 1 (Proxy-only). That is the hardest
protocol in the study, with test compounds carrying unseen scaffolds, and the
one where Graphormer is the best of all fifteen models (ROC-AUC 0.622, PR-AUC
0.786). Explaining a model in the setting where it leads is the defensible
pairing; on the random split at Stage 2 Graphormer is the weakest graph
transformer (0.591), so that run is deliberately not used here.

METHOD
------
SubgraphX (Yuan et al., ICML 2021) estimates each atom's Shapley value over
random permutations, then keeps the largest connected component of the
top-scoring atoms. The algorithm is the one from the original notebook,
unchanged in substance.

WHAT HAD TO CHANGE FOR THE CORRECTED MODELS
-------------------------------------------
The old explainer scored a candidate subgraph by calling the model with
(x, edge_index, batch, edge_attr) alone. Neither corrected architecture can be
scored that way:

  Graphormer consumes shortest-path-distance and edge-path encodings that are
  properties of the graph being scored, so deleting atoms changes them. They
  are therefore recomputed for every candidate subgraph; reusing the parent
  molecule's encodings would feed the model distances between atoms that are
  no longer connected.

  GraphGPS consumes a Laplacian positional encoding, which is likewise a
  property of the subgraph's own adjacency and is recomputed per candidate.

Scoring is done in eval mode, so GraphGPS's sign flipping is inactive and the
scores are deterministic.

Usage: python run_subgraphx.py [n_compounds|all] [n_rollouts]

Passing "all" explains every true positive rather than only the few drawn in
the figures, and writes a fragment-frequency table. The figures can only show a
handful of molecules, but the claim that the models learned chemically
meaningful alerts is better supported by how often a motif recurs across the
whole set than by six hand-picked examples.
Outputs (results/xai/):
    subgraphx_alerts.csv                 one row per explained compound
    <model>_tp<k>_<name>.svg             per-compound drawing
"""
import os
import random
import sys
import warnings; warnings.filterwarnings('ignore')

import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CONFIGS, data_dir
from graphormer_true import precompute_structural_encodings
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, subgraph as pyg_subgraph, to_dense_adj

SEED = 42
PROTO, UNIT, STAGE = 'scaffold_up', 0, 1
PE_DIM = 8
OUT = os.path.join(HERE, 'results', 'xai')
STATES = os.path.join(HERE, 'results', PROTO, 'states')


def compute_laplacian_pe(data, k=PE_DIM):
    """Same construction as DILI_assay_link/features.compute_laplacian_pe."""
    n = data.num_nodes
    if n <= 1:
        return torch.zeros(n, k)
    ei, ew = get_laplacian(data.edge_index, normalization='sym', num_nodes=n)
    L = to_dense_adj(ei, edge_attr=ew, max_num_nodes=n)[0]
    try:
        vals, vecs = torch.linalg.eigh(L)
    except Exception:
        return torch.zeros(n, k)
    pe = vecs[:, 1:k + 1]
    if pe.size(1) < k:
        pe = torch.cat([pe, torch.zeros(n, k - pe.size(1))], dim=1)
    return pe


def build_model(family, device='cpu'):
    import json
    cfg = json.load(open(f'{CONFIGS}/{family}_s1_winning_config.json'))
    if family == 'Graphormer':
        from graphormer_true import TrueGraphormer
        from gnn_model import ATOM_DIM
        m = TrueGraphormer(ATOM_DIM, hidden=cfg['hidden'], num_heads=cfg['num_heads'],
                           num_layers=cfg['num_layers'], dropout=cfg['dropout'])
    else:
        from gps_model import GraphGPSModel
        from gnn_model import ATOM_DIM
        m = GraphGPSModel(ATOM_DIM, hidden=cfg['hidden'], num_heads=cfg['num_heads'],
                          num_layers=cfg['num_layers'], dropout=cfg['dropout'])
    w = f'{STATES}/{family}_s{STAGE}_u{UNIT}.pt'
    if not os.path.exists(w):
        raise SystemExit(f'missing weights {w} -- run run_xai_models.sh first')
    m.load_state_dict(torch.load(w, map_location=device, weights_only=False))
    return m.to(device).eval()


@torch.no_grad()
def score_subgraph(model, family, x, edge_index, edge_attr, nodes, device='cpu'):
    """P(DILI) for the induced subgraph on `nodes`, encodings recomputed."""
    if not nodes:
        return 0.0
    idx = torch.tensor(sorted(nodes), dtype=torch.long)
    ei, ea = pyg_subgraph(idx, edge_index, edge_attr=edge_attr,
                          relabel_nodes=True, num_nodes=x.size(0))
    sub = Data(x=x[idx], edge_index=ei, edge_attr=ea)
    batch = torch.zeros(sub.num_nodes, dtype=torch.long, device=device)
    if family == 'Graphormer':
        # spd / edge-path are properties of THIS subgraph, so recompute them
        spd, epath = precompute_structural_encodings(sub)
        logit = model(sub.x.to(device), sub.edge_index.to(device),
                      sub.edge_attr.to(device), batch,
                      spd.to(device), epath.to(device))
    else:
        pe = compute_laplacian_pe(sub)             # Laplacian PE for THIS subgraph
        logit = model(sub.x.to(device), sub.edge_index.to(device),
                      sub.edge_attr.to(device), batch, pe.to(device))
    return float(torch.sigmoid(logit).reshape(-1)[0])


def largest_connected(nodes, edge_index):
    s = set(nodes)
    adj = {n: [] for n in s}
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        if u in s and v in s:
            adj[u].append(v)
    seen, comps = set(), []
    for start in s:
        if start in seen:
            continue
        comp, stack = [], [start]
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k); comp.append(k); stack.extend(adj[k])
        comps.append(comp)
    return max(comps, key=len) if comps else list(nodes)


def explain(model, family, g, min_atoms=3, rollouts=20, device='cpu'):
    """Monte-Carlo Shapley over atom permutations; returns a 0/1 atom mask."""
    x, ei, ea = g.x.cpu(), g.edge_index.cpu(), g.edge_attr.cpu()
    n = x.size(0)
    if n <= min_atoms:
        return torch.ones(n)
    shap = torch.zeros(n)
    for _ in range(rollouts):
        perm = list(range(n)); random.shuffle(perm)
        coalition, prev = [], 0.0
        for node in perm:
            coalition.append(node)
            sc = score_subgraph(model, family, x, ei, ea, coalition, device)
            shap[node] += (sc - prev) / rollouts
            prev = sc
    order = shap.argsort(descending=True)
    k = max(min_atoms, int((shap > shap.median()).sum()))
    keep = largest_connected(order[:k].tolist(), ei)
    mask = torch.zeros(n); mask[keep] = 1.0
    return mask


def draw(smiles, mask, path):
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atoms = [i for i, v in enumerate(mask.tolist()) if v > 0 and i < mol.GetNumAtoms()]
    acol = {i: (0.90, 0.20, 0.20) for i in atoms}
    bonds, bcol = [], {}
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms:
            bonds.append(b.GetIdx()); bcol[b.GetIdx()] = (0.90, 0.20, 0.20)
    d = rdMolDraw2D.MolDraw2DSVG(400, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atoms,
                                       highlightAtomColors=acol,
                                       highlightBonds=bonds,
                                       highlightBondColors=bcol)
    d.FinishDrawing()
    open(path, 'w').write(d.GetDrawingText())
    from rdkit.Chem import MolFragmentToSmiles
    try:
        return MolFragmentToSmiles(mol, atoms, bondsToUse=bonds) or None
    except Exception:
        return None


@torch.no_grad()
def predict_all(model, family, graphs, device='cpu'):
    out = []
    for g in graphs:
        batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
        if family == 'Graphormer':
            logit = model(g.x.to(device), g.edge_index.to(device),
                          g.edge_attr.to(device), batch,
                          g.spd_flat.to(device), g.edgepath_flat.to(device))
        else:
            logit = model(g.x.to(device), g.edge_index.to(device),
                          g.edge_attr.to(device), batch, g.pe.to(device))
        out.append(float(torch.sigmoid(logit).reshape(-1)[0]))
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

    rows, preds, models = [], {}, {}
    for family in ('Graphormer', 'GraphGPS'):
        model = build_model(family); models[family] = model
        p = predict_all(model, family, graphs)
        tp_all = np.where((p >= 0.5) & (y == 1))[0]
        preds[family] = (p, tp_all)
        tp = tp_all[np.argsort(-p[tp_all])][:n_top]
        t_fam = time.time()
        print(f'{family}: {len(tp_all)} true positives, '
              f'explaining the top {len(tp)}', flush=True)
        for rank, i in enumerate(tp, 1):
            mask = explain(model, family, graphs[i], rollouts=rollouts)
            svg = f'{OUT}/{family.lower()}_tp{rank}.svg'
            frag = draw(smi[i], mask, svg)
            rows.append(dict(model=family, rank=rank, inchikey=keys[i],
                             smiles=smi[i], prob=round(p[i], 4),
                             n_atoms=int(graphs[i].num_nodes),
                             n_highlighted=int(mask.sum()), fragment=frag,
                             svg=os.path.basename(svg),
                             mask=' '.join(str(int(v)) for v in mask.tolist())))
            if not explain_all:
                print(f'  {rank}. p={p[i]:.3f}  {int(mask.sum())}/{graphs[i].num_nodes}'
                      f' atoms  {frag}', flush=True)
            elif rank % 25 == 0:
                print(f'    ...{rank}/{len(tp)}', flush=True)
        if explain_all:
            print(f'  explained {len(tp)} compounds in {time.time()-t_fam:.0f}s', flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/subgraphx_alerts.csv', index=False)
    # ---- compounds both models call hepatotoxic ---------------------------
    # The consensus alert is the INTERSECTION of the two explanations: atoms
    # both architectures find important. Where that intersection is empty the
    # two models disagree on the motif, and the union is kept instead so the
    # panel still shows what each considered relevant.
    both = set(preds['Graphormer'][1]) & set(preds['GraphGPS'][1])
    agree = sorted(both, key=lambda i: -(preds['Graphormer'][0][i] + preds['GraphGPS'][0][i]))
    print(f'\n{len(both)} compounds are true positives for both models; '
          f'explaining the top {min(n_top, len(agree))}', flush=True)
    arows = []
    for rank, i in enumerate(agree[:n_top], 1):
        masks = {f: explain(models[f], f, graphs[i], rollouts=rollouts)
                 for f in ('Graphormer', 'GraphGPS')}
        inter = (masks['Graphormer'] * masks['GraphGPS'])
        mode = 'intersection'
        if inter.sum() == 0:
            inter = ((masks['Graphormer'] + masks['GraphGPS']) > 0).float()
            mode = 'union (no overlap)'
        svg = f'{OUT}/consensus_tp{rank}.svg'
        frag = draw(smi[i], inter, svg)
        arows.append(dict(model='Consensus', rank=rank, inchikey=keys[i],
                          smiles=smi[i],
                          p_graphormer=round(float(preds['Graphormer'][0][i]), 4),
                          p_graphgps=round(float(preds['GraphGPS'][0][i]), 4),
                          prob=round(float((preds['Graphormer'][0][i]
                                            + preds['GraphGPS'][0][i]) / 2), 4),
                          n_atoms=int(graphs[i].num_nodes),
                          n_highlighted=int(inter.sum()), overlap=mode,
                          fragment=frag, svg=os.path.basename(svg),
                          mask=' '.join(str(int(v)) for v in inter.tolist())))
        print(f'  {rank}. Graphormer {preds["Graphormer"][0][i]:.3f} / '
              f'GraphGPS {preds["GraphGPS"][0][i]:.3f}  '
              f'{int(inter.sum())}/{graphs[i].num_nodes} atoms [{mode}]  {frag}', flush=True)
    pd.DataFrame(arows).to_csv(f'{OUT}/subgraphx_agreement.csv', index=False)

    if explain_all:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')

        def canon(f):
            if not isinstance(f, str) or not f:
                return None
            m = Chem.MolFromSmiles(f, sanitize=False)
            try:
                return Chem.MolToSmiles(m) if m is not None else f
            except Exception:
                return f

        df = pd.DataFrame(rows)
        df['frag_canon'] = df.fragment.map(canon)
        tab = (df.dropna(subset=['frag_canon'])
                 .groupby(['frag_canon', 'model']).size().unstack(fill_value=0))
        for c in ('Graphormer', 'GraphGPS'):
            if c not in tab:
                tab[c] = 0
        tab['total'] = tab['Graphormer'] + tab['GraphGPS']
        tab['both_models'] = (tab['Graphormer'] > 0) & (tab['GraphGPS'] > 0)
        tab.sort_values(['both_models', 'total'], ascending=False).to_csv(
            f'{OUT}/subgraphx_fragment_frequency.csv')
        n_both = int(tab.both_models.sum())
        print(f'\n{len(tab)} distinct fragments recovered; {n_both} found by both models')
        print(tab.sort_values(['both_models', 'total'], ascending=False).head(12).to_string())
    print(f'\n-> {OUT}/subgraphx_alerts.csv')


if __name__ == '__main__':
    main()
