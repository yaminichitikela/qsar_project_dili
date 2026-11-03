"""Shared Graph Transformer architectures (Graphormer, GraphGPS) + train/eval
helpers. Ported from qsar_prediction/caspase/QSAR_Caspase_GraphTransformers.ipynb
(cells 18-19, 23), kept byte-for-byte identical in model definitions and
hyperparameter grids -- only the surrounding data pipeline changed (this repo's
cleaned summary-assay CSVs + 8:1:1/LOO/scaffold splits, not the notebook's
single 80/20/20 split).
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, GINEConv, GPSConv, global_mean_pool
from torch_geometric.utils import degree as pyg_degree, get_laplacian, to_dense_adj
from sklearn.metrics import roc_auc_score

SEED = 42
ATOM_DIM = 27
BOND_DIM = 6
PE_DIM = 8   # Laplacian eigenvectors for GraphGPS


# ── Graphormer ────────────────────────────────────────────────────────────────
class GraphormerLayer(nn.Module):
    """Graphormer block: TransformerConv -> LN + residual -> FFN -> LN + residual."""

    def __init__(self, hidden, num_heads, edge_dim=BOND_DIM, dropout=0.3):
        super().__init__()
        assert hidden % num_heads == 0
        head_dim = hidden // num_heads
        self.attn = TransformerConv(hidden, head_dim, heads=num_heads,
                                     concat=True, edge_dim=edge_dim,
                                     dropout=dropout, beta=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.norm2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        x = self.norm1(x + self.drop(self.attn(x, edge_index, edge_attr)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class GraphormerModel(nn.Module):
    """Graphormer-inspired graph transformer for binary molecular classification."""

    def __init__(self, in_dim=ATOM_DIM, hidden=128, num_heads=4,
                 num_layers=3, dropout=0.3):
        super().__init__()
        torch.manual_seed(SEED)
        self.node_emb = nn.Linear(in_dim, hidden)
        self.degree_emb = nn.Embedding(100, hidden)  # centrality encoding
        self.layers = nn.ModuleList([
            GraphormerLayer(hidden, num_heads, BOND_DIM, dropout)
            for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_emb(x)
        deg = pyg_degree(edge_index[0], num_nodes=x.size(0)).long().clamp(max=99)
        h = h + self.degree_emb(deg)
        h = F.relu(h)
        h = self.drop(h)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        h = global_mean_pool(h, batch)
        return self.head(h).squeeze(-1)


# ── GraphGPS ──────────────────────────────────────────────────────────────────
class GraphGPSModel(nn.Module):
    """GraphGPS: Laplacian PE + local GINEConv + global MultiheadAttention.
    Reference: Rampasek et al., NeurIPS 2022 (arXiv:2205.12454)"""

    def __init__(self, in_dim=ATOM_DIM, hidden=128, pe_dim=PE_DIM,
                 num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        torch.manual_seed(SEED)
        assert hidden % num_heads == 0
        self.node_emb = nn.Linear(in_dim, hidden)
        self.pe_emb = nn.Linear(pe_dim, hidden)
        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.GELU(),
                nn.Linear(hidden * 2, hidden))
            conv = GINEConv(mlp, edge_dim=BOND_DIM)
            gps = GPSConv(channels=hidden, conv=conv, heads=num_heads,
                          dropout=dropout, norm='layer_norm', attn_type='multihead')
            self.gps_layers.append(gps)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x, edge_index, edge_attr, batch, pe):
        h = self.node_emb(x) + self.pe_emb(pe)
        h = F.relu(h)
        h = self.drop(h)
        for gps in self.gps_layers:
            h = gps(h, edge_index, batch, edge_attr=edge_attr)
        h = global_mean_pool(h, batch)
        return self.head(h).squeeze(-1)


def compute_laplacian_pe(data, k=PE_DIM):
    """Adds a [num_nodes, k] Laplacian positional encoding to a PyG Data object.
    Only needed for GraphGPS -- Graphormer uses degree-based centrality encoding
    instead and reads the plain graphs built by prepare_split.py directly."""
    n = data.num_nodes
    if n <= 2:
        return torch.zeros(n, k)
    ei, ew = get_laplacian(data.edge_index, normalization='sym', num_nodes=n)
    L = to_dense_adj(ei, edge_attr=ew, max_num_nodes=n).squeeze(0).numpy()
    try:
        import numpy.linalg as nla
        _, vecs = nla.eigh(L)
        num_take = min(k, n - 1)
        pe = torch.from_numpy(vecs[:, 1:1 + num_take]).float()
        if pe.shape[1] < k:
            pe = F.pad(pe, (0, k - pe.shape[1]))
    except Exception:
        pe = torch.zeros(n, k)
    return pe


def gt_train(model, optimizer, loader, criterion, device, has_pe=False):
    model.train()
    total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logit = (model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe)
                  if has_pe else
                  model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
        loss = criterion(logit, batch.y)
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


@torch.no_grad()
def gt_eval(model, loader, device, has_pe=False):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logit = (model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe)
                  if has_pe else
                  model(batch.x, batch.edge_index, batch.edge_attr, batch.batch))
        all_logits.append(logit.cpu())
        all_labels.append(batch.y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    auc = roc_auc_score(labels, probs)
    return probs, labels, auc


# Same hyperparameter grid as QSAR_Caspase_GraphTransformers.ipynb cell 23
GT_CONFIGS = {
    'Graphormer': [
        {'hidden': 128, 'num_heads': 4, 'num_layers': 3, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden': 256, 'num_heads': 4, 'num_layers': 3, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden': 128, 'num_heads': 8, 'num_layers': 4, 'dropout': 0.4, 'lr': 5e-4},
    ],
    'GraphGPS': [
        {'hidden': 128, 'num_heads': 4, 'num_layers': 3, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden': 256, 'num_heads': 4, 'num_layers': 3, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden': 128, 'num_heads': 8, 'num_layers': 4, 'dropout': 0.4, 'lr': 5e-4},
    ],
}

MAX_EPOCHS = 100
PATIENCE = 20
