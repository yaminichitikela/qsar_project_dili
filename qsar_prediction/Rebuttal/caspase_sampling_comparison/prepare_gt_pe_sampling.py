"""Adds Laplacian positional encodings (needed by GraphGPS) to one variant's
graphs. Self-contained in this folder -- no dependency on either dataset's
wahab_jobs/.

Usage: python prepare_gt_pe_sampling.py <hepg2|cho_k1> <v1_ratio_1to5|v2_similarity|v3_diversity>
(not v4_smote -- that variant has no graphs, classical models only)
"""
import os
import sys
import warnings; warnings.filterwarnings('ignore')

import torch

from common import CASPASE_DATASETS, DATA_ROOT
from _prepare_gt_pe_ref import add_pe

VALID_VARIANTS = ['v1_ratio_1to5', 'v2_similarity', 'v3_diversity']

if len(sys.argv) != 3 or sys.argv[1] not in CASPASE_DATASETS or sys.argv[2] not in VALID_VARIANTS:
    print(f'Usage: python {sys.argv[0]} <{"|".join(CASPASE_DATASETS)}> <{"|".join(VALID_VARIANTS)}>')
    sys.exit(1)
DATASET_NAME, VARIANT = sys.argv[1], sys.argv[2]

DATA_DIR = os.path.join(DATA_ROOT, DATASET_NAME, VARIANT)

for split in ['train', 'val', 'test']:
    graphs = torch.load(os.path.join(DATA_DIR, f'graphs_{split}.pt'), weights_only=False)
    graphs_pe = add_pe(graphs)
    torch.save(graphs_pe, os.path.join(DATA_DIR, f'graphs_{split}_gps.pt'))
    print(f'{split}: {len(graphs_pe):,} graphs, Laplacian PE (dim=8) added')

print(f'\nSaved -> {DATA_DIR}/graphs_{{train,val,test}}_gps.pt  [{DATASET_NAME}/{VARIANT}]')
