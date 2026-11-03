"""Builds the bond-oriented graph view for each LOO rep, derived directly
from the atom-graphs already built by prepare_splits.py -- no RDKit
re-parsing needed (see prepare_mvgnn_bonds.py for the construction).

Requires prepare_splits.py to have been run first (loads from ./data_random_repeat/).

Usage: python prepare_mvgnn_bonds_rr.py
Outputs (written to ./data_random_repeat/repeat_<i>/):
    graphs_train_bond.pt, graphs_val_bond.pt, graphs_test_bond.pt
"""
import os
import warnings; warnings.filterwarnings('ignore')

import torch

from prepare_mvgnn_bonds import atom_graph_to_bond_graph

N_REPEATS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_random_repeat')


def main():
    for rep in range(N_REPEATS):
        rep_dir = os.path.join(DATA_DIR, f'repeat_{rep}')
        for split in ['train', 'val', 'test']:
            graphs = torch.load(os.path.join(rep_dir, f'graphs_{split}.pt'), weights_only=False)
            bond_graphs = [atom_graph_to_bond_graph(g) for g in graphs]
            torch.save(bond_graphs, os.path.join(rep_dir, f'graphs_{split}_bond.pt'))
        print(f'Repeat {rep}: bond-graphs built for train/val/test')

    print(f'\nSaved -> {DATA_DIR}/repeat_<0-4>/graphs_{{train,val,test}}_bond.pt')


if __name__ == '__main__':
    main()
