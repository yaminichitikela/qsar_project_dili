"""GraphGPS with a corrected positional encoding, for the DILI-linking experiment.

WHAT WAS WRONG
--------------
The GraphGPS architecture in Summary_1_models/modeling/wahab_jobs/gt_model.py is
faithful to Rampasek et al. (NeurIPS 2022): GPSConv already combines a local
GINEConv with global multi-head attention, which is the GPS recipe. The problem
is the positional encoding.

Laplacian eigenvectors are only defined up to sign -- both +v and -v are valid
eigenvectors of the same eigenvalue -- and numpy.linalg.eigh returns whichever
one its algorithm happens to produce. The existing pipeline computes the PE once
in prepare_gt_pe.py, caches it into graphs_*_gps.pt, and trains on it as if the
sign carried meaning. The model can then fit sign-specific patterns that will not
transfer to a new molecule whose eigendecomposition comes out flipped, and the
flip is essentially arbitrary across chemically similar molecules.

The GPS paper handles this with random sign flipping as a training-time
augmentation. That is what this file adds: on every forward pass during training,
each graph's PE columns are multiplied by independent random +/-1. The model is
therefore forced to learn sign-invariant use of the encoding. At eval time the
signs are left alone, so predictions are deterministic.

Everything else (GINEConv + GPSConv stack, dimensions, readout) is kept identical
to the existing implementation, so this is a correctness fix rather than a
redesign.

SCOPE: used only by the DILI-linking experiment in this folder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GPSConv, global_mean_pool

SEED = 42
ATOM_DIM = 27
BOND_DIM = 6
PE_DIM = 8


def random_sign_flip(pe, batch, generator=None):
    """Multiplies each graph's PE columns by independent random +/-1.

    pe:    [num_nodes, PE_DIM]
    batch: [num_nodes] graph assignment
    The sign is drawn per (graph, eigenvector) and broadcast to that graph's
    nodes, which is what keeps the encoding internally consistent within a
    molecule -- flipping per node would destroy the relative information the
    encoding carries.
    """
    n_graphs = int(batch.max()) + 1 if batch.numel() else 1
    signs = torch.randint(0, 2, (n_graphs, pe.size(1)), device=pe.device,
                          generator=generator, dtype=pe.dtype) * 2 - 1
    return pe * signs[batch]


class GraphGPSModel(nn.Module):
    """GraphGPS: Laplacian PE (sign-flip augmented) + local GINEConv + global
    multi-head attention.

    extra_dim > 0 appends that many molecule-level scalars (the two in vitro
    HepG2 assay predictions) to the pooled graph vector before the head.
    """

    def __init__(self, in_dim=ATOM_DIM, hidden=128, pe_dim=PE_DIM, num_heads=4,
                 num_layers=3, dropout=0.3, extra_dim=0, sign_flip=True):
        super().__init__()
        torch.manual_seed(SEED)
        assert hidden % num_heads == 0
        self.sign_flip = sign_flip
        self.node_emb = nn.Linear(in_dim, hidden)
        self.pe_emb = nn.Linear(pe_dim, hidden)
        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.GELU(),
                nn.Linear(hidden * 2, hidden))
            conv = GINEConv(mlp, edge_dim=BOND_DIM)
            self.gps_layers.append(
                GPSConv(channels=hidden, conv=conv, heads=num_heads,
                        dropout=dropout, norm='layer_norm', attn_type='multihead'))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + extra_dim, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x, edge_index, edge_attr, batch, pe, extra=None):
        if self.sign_flip and self.training:
            pe = random_sign_flip(pe, batch)
        h = self.node_emb(x) + self.pe_emb(pe)
        h = self.drop(F.relu(h))
        for gps in self.gps_layers:
            h = gps(h, edge_index, batch, edge_attr=edge_attr)
        h = global_mean_pool(h, batch)
        if extra is not None:
            h = torch.cat([h, extra.view(h.size(0), -1)], dim=-1)
        return self.head(h).squeeze(-1)
