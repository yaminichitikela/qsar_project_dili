"""Builds the bond-graph view (needed by MV-GNN-cross) for one variant.
Self-contained in this folder -- no dependency on either dataset's
wahab_jobs/.

Usage: python prepare_mvgnn_bonds_sampling.py <hepg2|cho_k1> <v1_ratio_1to5|v2_similarity|v3_diversity>
(not v4_smote -- that variant has no graphs, classical models only)
"""
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import torch

from common import CASPASE_DATASETS, DATA_ROOT
from _prepare_mvgnn_bonds_ref import atom_graph_to_bond_graph

VALID_VARIANTS = ['v1_ratio_1to5', 'v2_similarity', 'v3_diversity']

if len(sys.argv) != 3 or sys.argv[1] not in CASPASE_DATASETS or sys.argv[2] not in VALID_VARIANTS:
    print(f'Usage: python {sys.argv[0]} <{"|".join(CASPASE_DATASETS)}> <{"|".join(VALID_VARIANTS)}>')
    sys.exit(1)
DATASET_NAME, VARIANT = sys.argv[1], sys.argv[2]

DATA_DIR = os.path.join(DATA_ROOT, DATASET_NAME, VARIANT)

for split in ['train', 'val', 'test']:
    graphs = torch.load(os.path.join(DATA_DIR, f'graphs_{split}.pt'), weights_only=False)
    bond_graphs = [atom_graph_to_bond_graph(g) for g in graphs]
    torch.save(bond_graphs, os.path.join(DATA_DIR, f'graphs_{split}_bond.pt'))
    print(f'{split}: {len(bond_graphs):,} bond-graphs built')

print(f'\nSaved -> {DATA_DIR}/graphs_{{train,val,test}}_bond.pt  [{DATASET_NAME}/{VARIANT}]')
