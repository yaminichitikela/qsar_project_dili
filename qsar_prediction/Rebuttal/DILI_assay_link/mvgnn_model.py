"""MV-GNN^cross, ported from Ma et al. (deeplearningpaper.pdf), Figures 2-4 and
Equations 1-4 -- same architecture as Rebuttal/DILI_mvgnn/modeling/wahab_jobs/
mvgnn_model.py, adapted to this pipeline's existing 27-dim atom / 6-dim bond
features (ATOM_DIM/BOND_DIM from gnn_model.py) instead of the DILI notebook's
23/6-dim scheme, so MV-GNN sees the same inputs as GCN/GAT/GIN/SAGE/
Graphormer/GraphGPS already do in this pipeline.

  - Atom-oriented GNN + Bond-oriented GNN: two views of the same molecule
    (Eq. 1: generalized message passing). Both streams share the same layer
    class (ViewGNN), since the equation is entity-agnostic -- only the input
    graph differs (atom-graph from prepare_split.py vs. the bond-graph built
    by prepare_mvgnn_bonds.py).
  - Self-attentive readout (Eq. 2): single attention head, via PyG's
    AttentionalAggregation(gate_nn).
  - Two prediction heads (a, b); final prediction = average of probabilities.
  - Loss (Eq. 3/4): L = L_pred + lambda * L_dis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation
from sklearn.metrics import roc_auc_score

SEED = 42
ATOM_DIM = 27
BOND_DIM = 6


class PairedGraphDataset(Dataset):
    """Zips each molecule's atom-oriented and bond-oriented graph together so
    a single shuffle/batch always keeps both views aligned to the same
    molecule."""

    def __init__(self, atom_graphs, bond_graphs):
        assert len(atom_graphs) == len(bond_graphs)
        self.atom_graphs = atom_graphs
        self.bond_graphs = bond_graphs

    def __len__(self):
        return len(self.atom_graphs)

    def __getitem__(self, idx):
        return self.atom_graphs[idx], self.bond_graphs[idx]


def paired_collate(batch_list):
    atom_list, bond_list = zip(*batch_list)
    return Batch.from_data_list(list(atom_list)), Batch.from_data_list(list(bond_list))


def make_paired_loader(atom_graphs, bond_graphs, batch_size, shuffle):
    ds = PairedGraphDataset(atom_graphs, bond_graphs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=paired_collate)


class ViewGNN(nn.Module):
    """One message-passing view (atom-oriented or bond-oriented)."""

    def __init__(self, in_dim, edge_dim, hidden=128, n_layers=3, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.bn0 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.edge_projs = nn.ModuleList()
        for _ in range(n_layers):
            self.edge_projs.append(nn.Linear(edge_dim, hidden))
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
            self.convs.append(GINEConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.readout = AttentionalAggregation(
            gate_nn=nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)))

    def forward(self, x, edge_index, edge_attr, batch):
        h = F.relu(self.bn0(self.proj(x)))
        h = self.drop(h)
        for conv, bn, eproj in zip(self.convs, self.bns, self.edge_projs):
            e = eproj(edge_attr)
            h_res = h
            h = conv(h, edge_index, e)
            h = F.relu(bn(h))
            h = self.drop(h)
            h = h + h_res
        return self.readout(h, batch)


class MVGNNCross(nn.Module):
    def __init__(self, atom_dim=ATOM_DIM, atom_edge_dim=BOND_DIM,
                 bond_dim=BOND_DIM, bond_edge_dim=ATOM_DIM,
                 hidden=128, n_layers=3, dropout=0.3, extra_dim=0):
        super().__init__()
        torch.manual_seed(SEED)
        self.atom_gnn = ViewGNN(atom_dim, atom_edge_dim, hidden, n_layers, dropout)
        self.bond_gnn = ViewGNN(bond_dim, bond_edge_dim, hidden, n_layers, dropout)
        self.head_a = nn.Sequential(
            nn.Linear(hidden + extra_dim, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
        self.head_b = nn.Sequential(
            nn.Linear(hidden + extra_dim, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, atom_batch, bond_batch, extra=None):
        va = self.atom_gnn(atom_batch.x, atom_batch.edge_index, atom_batch.edge_attr, atom_batch.batch)
        vb = self.bond_gnn(bond_batch.x, bond_batch.edge_index, bond_batch.edge_attr, bond_batch.batch)
        # both views see the in vitro features, so neither stream is privileged
        if extra is not None:
            e = extra.view(va.size(0), -1)
            va, vb = torch.cat([va, e], dim=-1), torch.cat([vb, e], dim=-1)
        logit_a = self.head_a(va).squeeze(-1)
        logit_b = self.head_b(vb).squeeze(-1)
        return logit_a, logit_b


def mvgnn_loss(logit_a, logit_b, y, pos_weight, lam):
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    l_pred = 0.5 * (bce(logit_a, y) + bce(logit_b, y))
    l_dis = F.mse_loss(torch.sigmoid(logit_a), torch.sigmoid(logit_b))
    return l_pred + lam * l_dis, l_pred, l_dis


def mvgnn_train_epoch(model, paired_loader, optimizer, pos_weight, lam, device):
    model.train()
    total, n = 0, 0
    for atom_batch, bond_batch in paired_loader:
        atom_batch = atom_batch.to(device)
        bond_batch = bond_batch.to(device)
        optimizer.zero_grad()
        logit_a, logit_b = model(atom_batch, bond_batch)
        loss, _, _ = mvgnn_loss(logit_a, logit_b, atom_batch.y, pos_weight, lam)
        loss.backward()
        optimizer.step()
        total += loss.item() * atom_batch.num_graphs
        n += atom_batch.num_graphs
    return total / n


@torch.no_grad()
def mvgnn_eval(model, paired_loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for atom_batch, bond_batch in paired_loader:
        atom_batch = atom_batch.to(device)
        bond_batch = bond_batch.to(device)
        logit_a, logit_b = model(atom_batch, bond_batch)
        prob = 0.5 * (torch.sigmoid(logit_a) + torch.sigmoid(logit_b))
        all_probs.append(prob.cpu())
        all_labels.append(atom_batch.y.cpu())
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    auc = roc_auc_score(labels, probs)
    return probs, labels, auc


# Same 3-config style as gnn_model.py / gt_model.py.
MVGNN_CONFIGS = [
    {'hidden': 128, 'n_layers': 3, 'dropout': 0.3, 'lr': 1e-3, 'lam': 1.0},
    {'hidden': 256, 'n_layers': 3, 'dropout': 0.3, 'lr': 1e-3, 'lam': 0.5},
    {'hidden': 128, 'n_layers': 4, 'dropout': 0.4, 'lr': 5e-4, 'lam': 1.5},
]

MAX_EPOCHS = 100
PATIENCE = 20
