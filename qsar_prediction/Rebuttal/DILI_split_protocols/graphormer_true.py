"""Faithful Graphormer (Ying et al., NeurIPS 2021, arXiv:2106.05234) for the
DILI-linking experiment.

WHY THIS FILE EXISTS
--------------------
The Graphormer used everywhere else in this project
(Summary_1_models/modeling/wahab_jobs/gt_model.py, ported from
QSAR_Caspase_GraphTransformers.ipynb) builds its attention from PyG's
TransformerConv over edge_index, so every "attention" step is restricted to
bonded neighbours -- a 3-layer model sees 3 bonds out. Its own docstring calls
it "Graphormer-inspired", and that is accurate: of Graphormer's three
structural encodings it implements only the centrality (degree) encoding.

The manuscript, however, states that the graph transformers work "by
incorporating global attention that allows every node to attend to every other
node simultaneously" and cites Ying et al. That claim does not hold for the
TransformerConv version. This file implements what the manuscript describes:

  1. Global all-pairs attention        -- dense [N+1, N+1] attention, not 1-hop
  2. Centrality encoding               -- learned degree embedding added to h_0
  3. Spatial encoding                  -- learned per-head bias indexed by the
                                          shortest-path distance between i and j
  4. Edge encoding                     -- learned per-head bias from the mean
                                          bond features along the shortest path
  5. [VNode] virtual node              -- extra token attending to every atom,
                                          its final state used as the readout
                                          (replaces global_mean_pool)

Molecules here are small (tens of atoms), so dense all-pairs attention is cheap
and none of the large-graph approximations are needed.

SCOPE: used only by the DILI-linking experiment in this folder. The existing
summary-assay and DILI tables are produced by the older architecture and are not
affected by this file.

BATCHING NOTE
-------------
The spatial/edge encodings are per-pair, i.e. [N, N] and [N, N, BOND_DIM] per
graph, which default PyG collation cannot concatenate (N varies). They are
therefore stored FLATTENED as [N*N] and [N*N, BOND_DIM] by
precompute_structural_encodings(); flat tensors concatenate along dim 0 like any
other node/edge attribute, and build_dense_bias() reshapes and pads them back to
[B, Nmax, Nmax] inside the model. Attribute names deliberately avoid the
substring "index" so PyG does not try to increment their values during collation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from torch_geometric.utils import to_dense_batch

SEED = 42
ATOM_DIM = 27
BOND_DIM = 6

# Shortest-path-distance buckets. 0..MAX_SPD are real distances (0 = self),
# then one bucket for "unreachable" (disconnected fragments, which do occur in
# salts/mixtures) and one for any pair involving the virtual node.
MAX_SPD = 20
SPD_UNREACHABLE = MAX_SPD + 1
SPD_VNODE = MAX_SPD + 2
N_SPD_BUCKETS = MAX_SPD + 3

MAX_DEGREE = 16


# ── preprocessing (CPU, run once per molecule in the prep job) ────────────────
def precompute_structural_encodings(data):
    """Returns (spd_flat, edgepath_flat) for one PyG Data object.

    spd_flat       : [N*N]            long, bucketed shortest-path distances
    edgepath_flat  : [N*N, BOND_DIM]  float, mean bond features along one
                                      shortest path between each pair
    """
    n = int(data.num_nodes)
    ei = data.edge_index.numpy()
    ea = data.edge_attr.numpy() if data.edge_attr is not None else np.zeros((ei.shape[1], BOND_DIM))

    adj = np.zeros((n, n), dtype=np.float64)
    adj[ei[0], ei[1]] = 1.0
    # bond features keyed by (i, j); graphs are stored with both directions
    bond_feat = {}
    for k in range(ei.shape[1]):
        bond_feat[(int(ei[0, k]), int(ei[1, k]))] = ea[k]

    dist, pred = shortest_path(adj, method='D', unweighted=True,
                               directed=False, return_predecessors=True)

    spd = np.full((n, n), SPD_UNREACHABLE, dtype=np.int64)
    finite = np.isfinite(dist)
    spd[finite] = np.minimum(dist[finite], MAX_SPD).astype(np.int64)

    edgepath = np.zeros((n, n, BOND_DIM), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j or not finite[i, j]:
                continue
            # walk predecessors from j back to i, averaging bond features
            feats, cur, guard = [], j, 0
            while cur != i and cur >= 0 and guard <= MAX_SPD:
                prv = int(pred[i, cur])
                if prv < 0:
                    break
                f = bond_feat.get((prv, cur))
                if f is None:
                    f = bond_feat.get((cur, prv))
                if f is not None:
                    feats.append(f)
                cur, guard = prv, guard + 1
            if feats:
                edgepath[i, j] = np.mean(feats, axis=0)

    return (torch.from_numpy(spd.reshape(-1)),
            torch.from_numpy(edgepath.reshape(n * n, BOND_DIM)))


def build_dense_bias(spd_flat, edgepath_flat, counts, n_max):
    """Reshapes the flattened per-graph encodings back into padded dense
    tensors. Returns (spd_dense [B, n_max, n_max] long,
    edgepath_dense [B, n_max, n_max, BOND_DIM] float).

    Padded positions get SPD_UNREACHABLE; they are masked out of the attention
    anyway, but a valid bucket index is required for the embedding lookup.

    The prep job stores spd as int16 and edgepath as float16 to halve disk size;
    both are widened here (embedding lookups need long, the bias arithmetic needs
    float32).
    """
    b = counts.numel()
    dev = spd_flat.device
    spd_flat = spd_flat.long()
    edgepath_flat = edgepath_flat.float()
    spd_dense = torch.full((b, n_max, n_max), SPD_UNREACHABLE, dtype=torch.long, device=dev)
    edge_dense = torch.zeros((b, n_max, n_max, BOND_DIM), dtype=torch.float32, device=dev)

    offset = 0
    for g in range(b):
        n = int(counts[g])
        size = n * n
        spd_dense[g, :n, :n] = spd_flat[offset:offset + size].view(n, n)
        edge_dense[g, :n, :n] = edgepath_flat[offset:offset + size].view(n, n, BOND_DIM)
        offset += size
    return spd_dense, edge_dense


# ── model ─────────────────────────────────────────────────────────────────────
class GraphormerAttention(nn.Module):
    """Dense multi-head self-attention over ALL node pairs, with an additive
    per-head bias (spatial + edge encodings) injected into the attention logits.
    """

    def __init__(self, hidden, num_heads, dropout):
        super().__init__()
        assert hidden % num_heads == 0
        self.h = num_heads
        self.dk = hidden // num_heads
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, key_mask):
        """x: [B, T, D]; bias: [B, heads, T, T]; key_mask: [B, T] True = real."""
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # [B, heads, T, dk]

        logits = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5) + bias
        # block attention TO padded tokens
        logits = logits.masked_fill(~key_mask[:, None, None, :], float('-inf'))
        attn = self.drop(torch.softmax(logits, dim=-1))
        return self.out((attn @ v).transpose(1, 2).reshape(b, t, d))


class GraphormerBlock(nn.Module):
    """Pre-norm Graphormer block: LN -> global attention -> residual,
    then LN -> FFN -> residual (the ordering used in the paper)."""

    def __init__(self, hidden, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = GraphormerAttention(hidden, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden * 2, hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, bias, key_mask):
        x = x + self.drop(self.attn(self.norm1(x), bias, key_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TrueGraphormer(nn.Module):
    """Graphormer with centrality, spatial and edge encodings, a [VNode] readout
    token, and global all-pairs attention.

    extra_dim > 0 appends that many molecule-level scalars (here: the two in
    vitro HepG2 assay predictions) to the readout vector before the head, which
    is how the "Structure + in vitro" variant consumes them.
    """

    def __init__(self, in_dim=ATOM_DIM, hidden=128, num_heads=4, num_layers=3,
                 dropout=0.3, extra_dim=0):
        super().__init__()
        torch.manual_seed(SEED)
        self.node_emb = nn.Linear(in_dim, hidden)
        self.degree_emb = nn.Embedding(MAX_DEGREE + 1, hidden)        # centrality
        self.vnode = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.vnode, std=0.02)

        self.spd_bias = nn.Embedding(N_SPD_BUCKETS, num_heads)        # spatial
        self.edge_bias = nn.Linear(BOND_DIM, num_heads, bias=False)   # edge
        nn.init.zeros_(self.spd_bias.weight)
        nn.init.zeros_(self.edge_bias.weight)

        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [GraphormerBlock(hidden, num_heads, dropout) for _ in range(num_layers)])
        self.norm_out = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + extra_dim, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x, edge_index, edge_attr, batch, spd_flat, edgepath_flat,
                extra=None):
        h, mask = to_dense_batch(self.node_emb(x), batch)              # [B,N,D], [B,N]
        b, n, _ = h.shape

        deg = torch.bincount(edge_index[0], minlength=x.size(0)).clamp(max=MAX_DEGREE)
        deg_dense, _ = to_dense_batch(self.degree_emb(deg), batch)
        h = self.in_drop(h + deg_dense)

        counts = torch.bincount(batch, minlength=b).cpu()
        spd, edgepath = build_dense_bias(spd_flat, edgepath_flat, counts, n)

        # [B, heads, N, N] additive attention bias
        bias = self.spd_bias(spd).permute(0, 3, 1, 2) + \
            self.edge_bias(edgepath).permute(0, 3, 1, 2)

        # prepend the virtual node: its pair bias uses the dedicated SPD bucket
        v_bias = self.spd_bias(torch.full((b, 1), SPD_VNODE, dtype=torch.long,
                                          device=h.device))            # [B,1,heads]
        v_row = v_bias.permute(0, 2, 1).unsqueeze(-1).expand(b, bias.size(1), 1, n + 1)
        v_col = v_bias.permute(0, 2, 1).unsqueeze(-2).expand(b, bias.size(1), n, 1)
        bias = torch.cat([v_col, bias], dim=-1)                        # [B,h,N,N+1]
        bias = torch.cat([v_row, bias], dim=-2)                        # [B,h,N+1,N+1]

        h = torch.cat([self.vnode.expand(b, 1, -1), h], dim=1)         # [B,N+1,D]
        key_mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=h.device),
                              mask], dim=1)

        for blk in self.blocks:
            h = blk(h, bias, key_mask)

        graph_repr = self.norm_out(h[:, 0])                            # [VNode] readout
        if extra is not None:
            graph_repr = torch.cat([graph_repr, extra.view(b, -1)], dim=-1)
        return self.head(graph_repr).squeeze(-1)
