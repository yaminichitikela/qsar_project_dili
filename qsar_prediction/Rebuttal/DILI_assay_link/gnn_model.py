"""Shared MolGNN architecture + train/eval helpers, same as qsar_mito.ipynb Step 6."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, global_mean_pool
from sklearn.metrics import roc_auc_score

ATOM_DIM = 27


class MolGNN(nn.Module):
    """GNN for molecular binary classification. GCN, GAT, GIN, GraphSAGE via
    conv_type. 3 conv layers with BatchNorm + residual connections."""

    def __init__(self, in_dim, hidden=128, n_layers=3, dropout=0.3,
                 conv_type='GCN', heads=4, extra_dim=0):
        super().__init__()
        self.conv_type = conv_type
        self.n_layers = n_layers
        self.drop = nn.Dropout(dropout)

        self.proj = nn.Linear(in_dim, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(n_layers):
            if conv_type == 'GCN':
                self.convs.append(GCNConv(hidden, hidden))
            elif conv_type == 'GAT':
                self.convs.append(GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout))
            elif conv_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden * 2), nn.BatchNorm1d(hidden * 2),
                    nn.ReLU(), nn.Linear(hidden * 2, hidden))
                self.convs.append(GINConv(mlp, train_eps=True))
            elif conv_type == 'SAGE':
                self.convs.append(SAGEConv(hidden, hidden))
            else:
                raise ValueError(f'Unknown conv_type: {conv_type}')
            self.bns.append(nn.BatchNorm1d(hidden))

        self.head = nn.Sequential(
            nn.Linear(hidden + extra_dim, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, x, edge_index, batch, extra=None):
        x = F.relu(self.bn0(self.proj(x)))
        x = self.drop(x)
        for conv, bn in zip(self.convs, self.bns):
            x_res = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.drop(x)
            x = x + x_res
        x = global_mean_pool(x, batch)
        if extra is not None:
            x = torch.cat([x, extra.view(x.size(0), -1)], dim=-1)
        return self.head(x).squeeze(-1)


def gnn_train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def gnn_eval(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        all_logits.append(model(batch.x, batch.edge_index, batch.batch).cpu())
        all_labels.append(batch.y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    auc = roc_auc_score(labels, probs)
    return probs, labels, auc


# Same hyperparameter grid as qsar_mito.ipynb Step 6, split per architecture
GNN_CONFIGS = {
    'GCN': [
        {'hidden': 128, 'n_layers': 3, 'dropout': 0.3, 'lr': 0.001},
        {'hidden': 256, 'n_layers': 3, 'dropout': 0.3, 'lr': 0.001},
        {'hidden': 128, 'n_layers': 4, 'dropout': 0.4, 'lr': 0.0005},
    ],
    'GAT': [
        {'hidden': 128, 'n_layers': 3, 'dropout': 0.3, 'lr': 0.001, 'heads': 4},
        {'hidden': 128, 'n_layers': 3, 'dropout': 0.4, 'lr': 0.0005, 'heads': 8},
        {'hidden': 256, 'n_layers': 3, 'dropout': 0.3, 'lr': 0.001, 'heads': 4},
    ],
    'GIN': [
        {'hidden': 128, 'n_layers': 3, 'dropout': 0.3, 'lr': 0.001},
        {'hidden': 256, 'n_layers': 3, 'dropout': 0.3, 'lr': 0.001},
        {'hidden': 128, 'n_layers': 4, 'dropout': 0.4, 'lr': 0.0005},
    ],
    'SAGE': [
        {'hidden': 128, 'n_layers': 3, 'dropout': 0.3, 'lr': 0.001},
        {'hidden': 256, 'n_layers': 3, 'dropout': 0.3, 'lr': 0.001},
        {'hidden': 128, 'n_layers': 4, 'dropout': 0.4, 'lr': 0.0005},
    ],
}

MAX_EPOCHS = 100
PATIENCE = 20
