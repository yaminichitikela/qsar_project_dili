"""MV-GNN^cross, ported from Ma et al. (deeplearningpaper.pdf), Figures 2-4 and
Equations 1-4:
  - Atom-oriented GNN + Bond-oriented GNN: two views of the same molecule,
    entity o = atoms or bonds respectively (Eq. 1: generalized message
    passing -- m_o^(d+1) = sum_n A_d(h_n^d, mu_attached); h_o^(d+1) =
    U_d(h_o^d, m_o^(d+1))). Both streams share the same layer class here,
    since the equation is entity-agnostic -- only the input graph differs
    (atom-graph vs. the bond-graph built by prepare_dili_data.py).
  - Self-attentive readout (Eq. 2): S = softmax(W2 tanh(W1 H)), single
    attention head (the paper's r is unspecified, so r=1 here) -- implemented
    via PyG's AttentionalAggregation(gate_nn), which computes exactly this gating.
  - Two prediction heads (a, b), one per stream; final prediction = average
    of their probabilities (paper's Fig. 2 doesn't specify the combination
    rule).
  - Loss (Eq. 3/4, single-task simplification -- no missing-label mask
    needed since DILI is not augmented here): L = L_pred + lambda * L_dis,
    L_pred = mean(BCE_a, BCE_b), L_dis = MSE(sigmoid(logit_a), sigmoid(logit_b)).
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


class PairedGraphDataset(Dataset):
    """Zips each molecule's atom-oriented and bond-oriented graph together so
    a single shuffle/batch always keeps both views aligned to the same
    molecule -- independent DataLoaders over the two lists would shuffle
    differently and silently misalign atom/bond batches."""

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
    """One message-passing view (atom-oriented or bond-oriented). Generic:
    works for either entity type, since Eq. 1 doesn't distinguish them --
    only the (in_dim, edge_dim) of the graph it's fed changes."""

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
        return self.readout(h, batch)  # [num_graphs, hidden]


class MVGNNCross(nn.Module):
    def __init__(self, atom_dim, atom_edge_dim, bond_dim, bond_edge_dim,
                 hidden=128, n_layers=3, dropout=0.3):
        super().__init__()
        torch.manual_seed(SEED)
        self.atom_gnn = ViewGNN(atom_dim, atom_edge_dim, hidden, n_layers, dropout)
        self.bond_gnn = ViewGNN(bond_dim, bond_edge_dim, hidden, n_layers, dropout)
        self.head_a = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))
        self.head_b = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    def forward(self, atom_batch, bond_batch):
        va = self.atom_gnn(atom_batch.x, atom_batch.edge_index, atom_batch.edge_attr, atom_batch.batch)
        vb = self.bond_gnn(bond_batch.x, bond_batch.edge_index, bond_batch.edge_attr, bond_batch.batch)
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
    total = 0
    n = 0
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


# 3-config grid, same style as gnn_model.py / gt_model.py. Paper does not
# specify hidden/layers/lr/lambda -- these are reasonable defaults consistent
# with the rest of the pipeline, not values taken from the paper.
MVGNN_CONFIGS = [
    {'hidden': 128, 'n_layers': 3, 'dropout': 0.3, 'lr': 1e-3, 'lam': 1.0},
    {'hidden': 256, 'n_layers': 3, 'dropout': 0.3, 'lr': 1e-3, 'lam': 0.5},
    {'hidden': 128, 'n_layers': 4, 'dropout': 0.4, 'lr': 5e-4, 'lam': 1.0},
]

MAX_EPOCHS = 100
PATIENCE = 20
