"""Builds the bond-oriented graph view MV-GNN^cross needs, for every split unit
of one protocol, directly from that unit's atom graphs (see
prepare_mvgnn_bonds.py for the construction).

Usage: python prepare_mvgnn_bonds_protocol.py <loo|scaffold>
Outputs: graphs_{train,val,test}_bond.pt next to each unit's graphs_*.pt
"""
import os
import warnings; warnings.filterwarnings('ignore')

import torch

from prepare_mvgnn_bonds import atom_graph_to_bond_graph
from protocols import from_argv, unit_dir

P = from_argv()

for i in range(P['n']):
    d = unit_dir(P, i)
    for split in ['train', 'val', 'test']:
        graphs = torch.load(os.path.join(d, f'graphs_{split}.pt'), weights_only=False)
        torch.save([atom_graph_to_bond_graph(g) for g in graphs],
                   os.path.join(d, f'graphs_{split}_bond.pt'))
    print(f'{P["unit"]} {i}: bond-graphs built for train/val/test', flush=True)
