"""Adds Laplacian positional encodings to every split unit of one protocol, for
GraphGPS only (the corrected Graphormer computes its shortest-path encodings
in-job).

Usage: python prepare_gt_pe_protocol.py <scaffold>
Outputs: graphs_{train,val,test}_gps.pt next to each unit's graphs_*.pt
"""
import os
import warnings; warnings.filterwarnings('ignore')

import torch
from torch_geometric.data import Data

from gt_model import compute_laplacian_pe
from protocols import from_argv, unit_dir

P = from_argv()

for i in range(P['n']):
    d = unit_dir(P, i)
    for split in ['train', 'val', 'test']:
        graphs = torch.load(os.path.join(d, f'graphs_{split}.pt'), weights_only=False)
        out = [Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, y=g.y,
                    pe=compute_laplacian_pe(g)) for g in graphs]
        torch.save(out, os.path.join(d, f'graphs_{split}_gps.pt'))
    print(f'{P["unit"]} {i}: Laplacian PE added to train/val/test graphs', flush=True)
