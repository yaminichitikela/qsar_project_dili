"""Adds Laplacian positional encodings to the already-built 8:1:1 split graphs,
for GraphGPS only (Graphormer reuses data/graphs_*.pt directly -- it already
has edge_attr, built by prepare_split.py, and uses degree-based centrality
encoding instead of Laplacian PE).

Requires prepare_split.py to have been run first (loads from ./data/).

Usage: python prepare_gt_pe.py
Outputs (written to ./data/):
    graphs_train_gps.pt, graphs_val_gps.pt, graphs_test_gps.pt
"""
import os
import warnings; warnings.filterwarnings('ignore')

import torch
from torch_geometric.data import Data

from gt_model import compute_laplacian_pe

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def add_pe(graphs):
    out = []
    for d in graphs:
        pe = compute_laplacian_pe(d)
        out.append(Data(x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y, pe=pe))
    return out


def main():
    for split in ['train', 'val', 'test']:
        graphs = torch.load(os.path.join(DATA_DIR, f'graphs_{split}.pt'), weights_only=False)
        graphs_pe = add_pe(graphs)
        torch.save(graphs_pe, os.path.join(DATA_DIR, f'graphs_{split}_gps.pt'))
        print(f'{split}: {len(graphs_pe):,} graphs, Laplacian PE (dim=8) added')

    print(f'\nSaved -> {DATA_DIR}/graphs_{{train,val,test}}_gps.pt')


if __name__ == '__main__':
    main()
