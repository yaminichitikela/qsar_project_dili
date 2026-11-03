"""Builds the bond-oriented graph view needed by MV-GNN^cross, derived
directly from the atom-oriented graphs already built by prepare_split.py --
no RDKit/SMILES re-parsing, so it's guaranteed to describe the exact same
molecules/featurization already used by every other model in this pipeline.

Bond-graph construction (the paper's "line graph" view): each unique bond
becomes a node (feature = that bond's 6-dim edge_attr from the atom-graph);
two bond-nodes are connected if they share an atom, with the edge feature
being that shared atom's 27-dim atom feature.

Requires prepare_split.py to have been run first (loads from ./data/).

Usage: python prepare_mvgnn_bonds.py
Outputs (written to ./data/):
    graphs_train_bond.pt, graphs_val_bond.pt, graphs_test_bond.pt
"""
import os
import warnings; warnings.filterwarnings('ignore')

import torch
from torch_geometric.data import Data

from mvgnn_model import ATOM_DIM, BOND_DIM

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def atom_graph_to_bond_graph(data):
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    y = data.y

    ei = edge_index.t().tolist()
    ea = edge_attr.tolist()

    seen = {}
    bond_list = []  # (atom_i, atom_j, bond_feat)
    for (i, j), bf in zip(ei, ea):
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen[key] = len(bond_list)
            bond_list.append((i, j, bf))

    if len(bond_list) == 0:
        bx = torch.zeros((1, BOND_DIM), dtype=torch.float)
        bei = torch.zeros((2, 0), dtype=torch.long)
        bea = torch.zeros((0, ATOM_DIM), dtype=torch.float)
        return Data(x=bx, edge_index=bei, edge_attr=bea, y=y)

    bx = torch.tensor([bf for _, _, bf in bond_list], dtype=torch.float)

    atom_to_bonds = {}
    for bi, (i, j, _) in enumerate(bond_list):
        atom_to_bonds.setdefault(i, []).append(bi)
        atom_to_bonds.setdefault(j, []).append(bi)

    edges, attrs = [], []
    for atom_idx, blist in atom_to_bonds.items():
        if len(blist) < 2:
            continue
        af = x[atom_idx].tolist()
        for a in range(len(blist)):
            for b in range(len(blist)):
                if a == b:
                    continue
                edges.append([blist[a], blist[b]])
                attrs.append(af)

    if edges:
        bei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        bea = torch.tensor(attrs, dtype=torch.float)
    else:
        bei = torch.zeros((2, 0), dtype=torch.long)
        bea = torch.zeros((0, ATOM_DIM), dtype=torch.float)

    return Data(x=bx, edge_index=bei, edge_attr=bea, y=y)


def main():
    for split in ['train', 'val', 'test']:
        graphs = torch.load(os.path.join(DATA_DIR, f'graphs_{split}.pt'), weights_only=False)
        bond_graphs = [atom_graph_to_bond_graph(g) for g in graphs]
        torch.save(bond_graphs, os.path.join(DATA_DIR, f'graphs_{split}_bond.pt'))
        print(f'{split}: {len(bond_graphs):,} bond-graphs built')

    print(f'\nSaved -> {DATA_DIR}/graphs_{{train,val,test}}_bond.pt')


if __name__ == '__main__':
    main()
