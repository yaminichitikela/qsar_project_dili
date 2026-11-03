"""Adds Laplacian positional encodings to each random-repeat split's graphs,
for GraphGPS only (Graphormer reuses data_random_repeat/repeat_<i>/graphs_*.pt
directly).

Requires prepare_random_repeat_splits.py to have been run first (loads from
./data_random_repeat/).

Usage: python prepare_gt_pe_random_repeat.py
Outputs (written to ./data_random_repeat/repeat_<i>/):
    graphs_train_gps.pt, graphs_val_gps.pt, graphs_test_gps.pt
"""
import os
import warnings; warnings.filterwarnings('ignore')

import torch
from torch_geometric.data import Data

from gt_model import compute_laplacian_pe

N_REPEATS = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_random_repeat')


def add_pe(graphs):
    out = []
    for d in graphs:
        pe = compute_laplacian_pe(d)
        out.append(Data(x=d.x, edge_index=d.edge_index, edge_attr=d.edge_attr, y=d.y, pe=pe))
    return out


def main():
    for rep in range(N_REPEATS):
        rep_dir = os.path.join(DATA_DIR, f'repeat_{rep}')
        for split in ['train', 'val', 'test']:
            graphs = torch.load(os.path.join(rep_dir, f'graphs_{split}.pt'), weights_only=False)
            graphs_pe = add_pe(graphs)
            torch.save(graphs_pe, os.path.join(rep_dir, f'graphs_{split}_gps.pt'))
        print(f'Repeat {rep}: Laplacian PE added to train/val/test graphs')

    print(f'\nSaved -> {DATA_DIR}/repeat_<0-4>/graphs_{{train,val,test}}_gps.pt')


if __name__ == '__main__':
    main()
