"""Builds the bond-oriented graph view for each LOO fold, derived directly
from the atom-graphs already built by prepare_loo_folds.py -- no RDKit
re-parsing needed (see prepare_mvgnn_bonds.py for the construction).

Requires prepare_loo_folds.py to have been run first (loads from ./data_loo/).

Usage: python prepare_mvgnn_bonds_loo.py
Outputs (written to ./data_loo/fold_<i>/):
    graphs_train_bond.pt, graphs_val_bond.pt, graphs_test_bond.pt
"""
import os
import warnings; warnings.filterwarnings('ignore')

import torch

from prepare_mvgnn_bonds import atom_graph_to_bond_graph

N_FOLDS = 10
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_loo')


def main():
    for fold in range(N_FOLDS):
        fold_dir = os.path.join(DATA_DIR, f'fold_{fold}')
        for split in ['train', 'val', 'test']:
            graphs = torch.load(os.path.join(fold_dir, f'graphs_{split}.pt'), weights_only=False)
            bond_graphs = [atom_graph_to_bond_graph(g) for g in graphs]
            torch.save(bond_graphs, os.path.join(fold_dir, f'graphs_{split}_bond.pt'))
        print(f'Fold {fold}: bond-graphs built for train/val/test')

    print(f'\nSaved -> {DATA_DIR}/fold_<0-9>/graphs_{{train,val,test}}_bond.pt')


if __name__ == '__main__':
    main()
