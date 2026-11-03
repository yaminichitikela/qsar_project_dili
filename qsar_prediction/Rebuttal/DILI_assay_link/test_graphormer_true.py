"""Correctness tests for graphormer_true.py before it goes to Wahab."""
import os
import sys
import numpy as np
import networkx as nx
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graphormer_true import (TrueGraphormer, precompute_structural_encodings,
                             ATOM_DIM, BOND_DIM, MAX_SPD, SPD_UNREACHABLE)

SMILES = [
    'CC(=O)Oc1ccccc1C(=O)O',                      # aspirin
    'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',               # caffeine
    'c1ccccc1',                                   # benzene
    'CCO',                                        # tiny
    'C',                                          # single atom, no bonds
    '[Na+].[Cl-]',                                # disconnected (salt)
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',                 # ibuprofen
]
rng = np.random.RandomState(0)


def mol_to_data(smi, perm=None):
    mol = Chem.MolFromSmiles(smi)
    n = mol.GetNumAtoms()
    order = list(range(n)) if perm is None else list(perm)
    inv = {old: new for new, old in enumerate(order)}
    # deterministic per-atom features so permutation tests are meaningful
    feats = np.stack([rng_for(smi, a) for a in range(n)])
    x = torch.tensor(feats[order], dtype=torch.float)
    src, dst, eattr = [], [], []
    for bnd in mol.GetBonds():
        i, j = inv[bnd.GetBeginAtomIdx()], inv[bnd.GetEndAtomIdx()]
        bf = rng_for_bond(smi, bnd.GetIdx())
        src += [i, j]; dst += [j, i]; eattr += [bf, bf]
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.tensor(np.array(eattr), dtype=torch.float) if eattr else torch.zeros(0, BOND_DIM)
    d = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor([0.0]))
    d.num_nodes = n
    spd, ep = precompute_structural_encodings(d)
    d.spd_flat, d.edgepath_flat = spd, ep
    return d, mol, order


_cache = {}
def rng_for(smi, a):
    key = ('a', smi, a)
    if key not in _cache:
        _cache[key] = np.random.RandomState(abs(hash(key)) % 2**31).rand(ATOM_DIM)
    return _cache[key]


def rng_for_bond(smi, b):
    key = ('b', smi, b)
    if key not in _cache:
        _cache[key] = np.random.RandomState(abs(hash(key)) % 2**31).rand(BOND_DIM)
    return _cache[key]


fails = []

# ── TEST 1: shortest-path distances match networkx ───────────────────────────
print('TEST 1  shortest-path distances vs networkx')
for smi in SMILES:
    d, mol, _ = mol_to_data(smi)
    n = d.num_nodes
    spd = d.spd_flat.view(n, n).numpy()
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])
    ref = dict(nx.all_pairs_shortest_path_length(g))
    bad = 0
    for i in range(n):
        for j in range(n):
            want = ref[i].get(j)
            want = SPD_UNREACHABLE if want is None else min(want, MAX_SPD)
            if spd[i, j] != want:
                bad += 1
    status = 'ok' if bad == 0 else f'{bad} MISMATCHES'
    print(f'   {smi:32s} n={n:2d}  {status}')
    if bad:
        fails.append(f'SPD mismatch for {smi}')

# ── TEST 2: 1-hop edge-path features equal the actual bond features ──────────
print('\nTEST 2  edge encoding at distance 1 equals the bond feature')
for smi in ['CCO', 'CC(=O)Oc1ccccc1C(=O)O']:
    d, mol, _ = mol_to_data(smi)
    n = d.num_nodes
    ep = d.edgepath_flat.view(n, n, BOND_DIM).numpy()
    ok = True
    for k in range(d.edge_index.size(1)):
        i, j = int(d.edge_index[0, k]), int(d.edge_index[1, k])
        if not np.allclose(ep[i, j], d.edge_attr[k].numpy(), atol=1e-6):
            ok = False
    print(f'   {smi:32s} {"ok" if ok else "MISMATCH"}')
    if not ok:
        fails.append(f'edge-path mismatch for {smi}')

# ── TEST 3: permutation invariance of the model output ───────────────────────
print('\nTEST 3  permutation invariance (atom reordering must not change output)')
model = TrueGraphormer(hidden=64, num_heads=4, num_layers=2, dropout=0.0).eval()
for smi in SMILES:
    d0, mol, _ = mol_to_data(smi)
    n = d0.num_nodes
    with torch.no_grad():
        b0 = next(iter(DataLoader([d0], batch_size=1)))
        out0 = model(b0.x, b0.edge_index, b0.edge_attr, b0.batch,
                     b0.spd_flat, b0.edgepath_flat).item()
    diffs = []
    for t in range(3):
        perm = np.random.RandomState(t).permutation(n)
        dp, _, _ = mol_to_data(smi, perm=perm)
        with torch.no_grad():
            bp = next(iter(DataLoader([dp], batch_size=1)))
            outp = model(bp.x, bp.edge_index, bp.edge_attr, bp.batch,
                         bp.spd_flat, bp.edgepath_flat).item()
        diffs.append(abs(out0 - outp))
    m = max(diffs)
    print(f'   {smi:32s} max|delta|={m:.2e}  {"ok" if m < 1e-4 else "NOT INVARIANT"}')
    if m >= 1e-4:
        fails.append(f'permutation variance for {smi}: {m:.2e}')

# ── TEST 4: batch invariance (padding must not leak between molecules) ───────
print('\nTEST 4  batch invariance (same prediction alone vs padded in a batch)')
datas = [mol_to_data(s)[0] for s in SMILES]
singles = []
for d in datas:
    with torch.no_grad():
        b = next(iter(DataLoader([d], batch_size=1)))
        singles.append(model(b.x, b.edge_index, b.edge_attr, b.batch,
                             b.spd_flat, b.edgepath_flat).item())
with torch.no_grad():
    bb = next(iter(DataLoader(datas, batch_size=len(datas), shuffle=False)))
    batched = model(bb.x, bb.edge_index, bb.edge_attr, bb.batch,
                    bb.spd_flat, bb.edgepath_flat).numpy()
delta = np.abs(np.array(singles) - batched)
for smi, s, b_, dd in zip(SMILES, singles, batched, delta):
    print(f'   {smi:32s} single={s:+.6f} batched={b_:+.6f} delta={dd:.2e}')
if delta.max() >= 1e-4:
    fails.append(f'batch leakage: max delta {delta.max():.2e}')
print(f'   max delta = {delta.max():.2e}  {"ok" if delta.max() < 1e-4 else "LEAKAGE"}')

# ── TEST 5: global attention actually reaches distant atoms ──────────────────
print('\nTEST 5  a far-away atom influences the prediction (true global attention)')
smi = 'CCCCCCCCCCCCCCCC'            # 16-carbon chain, diameter 15 >> 2 layers
d, mol, _ = mol_to_data(smi)
with torch.no_grad():
    b = next(iter(DataLoader([d], batch_size=1)))
    base = model(b.x, b.edge_index, b.edge_attr, b.batch, b.spd_flat, b.edgepath_flat).item()
    d2 = d.clone()
    d2.x = d2.x.clone(); d2.x[-1] += 5.0      # perturb only the far-end atom
    b2 = next(iter(DataLoader([d2], batch_size=1)))
    pert = model(b2.x, b2.edge_index, b2.edge_attr, b2.batch, b2.spd_flat, b2.edgepath_flat).item()
shift = abs(base - pert)
print(f'   chain n={d.num_nodes}, 2 layers, perturbing terminal atom: |delta|={shift:.3e}')
print(f'   {"ok (global attention reaches it)" if shift > 1e-5 else "NO EFFECT -- attention is local"}')
if shift <= 1e-5:
    fails.append('far atom had no influence; attention is not global')

# ── TEST 6: extra molecule-level features + gradient flow ───────────────────
print('\nTEST 6  in vitro extra features and gradient flow')
m2 = TrueGraphormer(hidden=64, num_heads=4, num_layers=2, dropout=0.1, extra_dim=2)
bb = next(iter(DataLoader(datas, batch_size=len(datas), shuffle=False)))
extra = torch.rand(len(datas), 2, requires_grad=False)
out = m2(bb.x, bb.edge_index, bb.edge_attr, bb.batch, bb.spd_flat, bb.edgepath_flat, extra=extra)
loss = torch.nn.functional.binary_cross_entropy_with_logits(out, torch.zeros_like(out))
loss.backward()
named = dict(m2.named_parameters())
grads = {k: (named[k].grad is not None and named[k].grad.abs().sum().item() > 0)
         for k in ['vnode', 'spd_bias.weight', 'edge_bias.weight', 'degree_emb.weight', 'node_emb.weight']}
print(f'   output shape {tuple(out.shape)} (expect ({len(datas)},))')
for k, v in grads.items():
    print(f'   grad flows to {k:22s} {"yes" if v else "NO"}')
if not all(grads.values()):
    fails.append(f'no gradient to: {[k for k,v in grads.items() if not v]}')
if out.shape != (len(datas),):
    fails.append('wrong output shape')

print('\n' + '=' * 60)
print('ALL TESTS PASSED' if not fails else 'FAILURES:\n  - ' + '\n  - '.join(fails))
sys.exit(1 if fails else 0)
