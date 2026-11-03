"""One training/prediction engine for all ten model families.

Both halves of this experiment call into here -- the assay side (to build the in
vitro features) and the DILI side (to compare Structure-only against
Structure + in vitro). Sharing the engine is deliberate: if the two sides trained
through separate code paths they could drift in early-stopping rule, metric, or
probability calibration, and the comparison would stop being about the features.

Every family is selected the same way: fit on train, score val ROC-AUC, keep the
best. Graph models additionally early-stop on val ROC-AUC and restore the best
epoch's weights. The test set is touched exactly once, after selection.

`extra` is the optional molecule-level feature block (here: the two in vitro
HepG2 predictions). It is appended to the pooled graph vector for graph models
and concatenated to the fingerprint for classical models, so "Structure-only" vs
"Structure + in vitro" differs only in whether it is passed.

LAZY IMPORTS
------------
torch, PyG and the model modules are imported inside the functions that need
them, not at module scope, so a classical-only job (RF/SVM/XGBoost) never loads
torch. That keeps those jobs light on Wahab, and on macOS it also avoids the
duplicate-OpenMP crash that happens when torch is loaded before xgboost.
"""
import copy

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

SEED = 42
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 20
GNN_FAMILIES = ('GCN', 'GAT', 'GIN', 'SAGE')


# ── data container ───────────────────────────────────────────────────────────
class SplitData:
    """Holds one repeat's splits as index arrays into shared, featurized arrays.

    graphs/bond_graphs are the full shared lists written by the prep job; idx maps
    a split name to positions in them. extra is either None or an
    [n_compounds, extra_dim] array aligned the same way.
    """

    def __init__(self, X, y, graphs, bond_graphs, idx, extra=None):
        self.X, self.y = X, y
        self.graphs, self.bond_graphs = graphs, bond_graphs
        self.idx = idx
        self.extra = extra

    @property
    def extra_dim(self):
        return 0 if self.extra is None else self.extra.shape[1]

    def fp(self, split):
        """Fingerprints, with the in vitro block appended when present."""
        X = self.X[self.idx[split]]
        if self.extra is None:
            return X
        return np.hstack([X, self.extra[self.idx[split]]])

    def labels(self, split):
        return self.y[self.idx[split]]

    def graph_list(self, split, bond=False):
        import torch
        from torch_geometric.data import Data
        src = self.bond_graphs if bond else self.graphs
        out = []
        for i in self.idx[split]:
            g = src[i]
            kw = dict(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr, y=g.y)
            for attr in ('pe', 'spd_flat', 'edgepath_flat'):
                if hasattr(g, attr) and getattr(g, attr) is not None:
                    kw[attr] = getattr(g, attr)
            if self.extra is not None:
                # one row per graph, so default collation stacks to [B, extra_dim]
                kw['extra'] = torch.tensor(self.extra[i], dtype=torch.float).view(1, -1)
            out.append(Data(**kw))
        return out


# ── classical ────────────────────────────────────────────────────────────────
def _make_classical(family, params, y_train, calibrate=True):
    if family == 'RF':
        return RandomForestClassifier(**params, class_weight='balanced',
                                      n_jobs=-1, random_state=SEED)
    if family == 'SVM':
        # probability=True fits an internal 5-fold Platt calibration on every
        # fit. During a hyperparameter search that cost is wasted: selection is
        # on ROC-AUC, which is invariant to any monotone rescaling of the score,
        # so decision_function() ranks identically to the calibrated
        # probability. Only the winning config is refitted calibrated, which is
        # exactly the trick classical_config.py uses in DILI_proxy_clean.
        return SVC(**params, kernel='rbf', class_weight='balanced',
                   probability=calibrate, random_state=SEED)
    if family == 'XGB':
        from xgboost import XGBClassifier
        pos = max(int((y_train == 1).sum()), 1)
        sw = float((y_train == 0).sum()) / pos
        return XGBClassifier(**params, scale_pos_weight=sw, eval_metric='logloss',
                             n_jobs=-1, random_state=SEED, verbosity=0)
    raise ValueError(family)


def _run_classical(family, params, data, predict_sets, extra_predict, calibrate=True):
    y_tr = data.labels('train')
    clf = _make_classical(family, params, y_tr, calibrate=calibrate)
    clf.fit(data.fp('train'), y_tr)

    def score(X):
        if calibrate or family != 'SVM':
            return clf.predict_proba(X)[:, 1]
        return clf.decision_function(X)        # ranks identically for ROC-AUC

    probs = {s: score(data.fp(s)) for s in predict_sets}
    for name, (X_ext, extra_block) in (extra_predict or {}).items():
        Xp = X_ext if extra_block is None else np.hstack([X_ext, extra_block])
        probs[name] = score(Xp)
    val_auc = roc_auc_score(data.labels('val'), probs['val'])
    return val_auc, probs, clf


# ── graph ────────────────────────────────────────────────────────────────────
def _build_graph_model(family, params, extra_dim, device):
    from gnn_model import MolGNN, ATOM_DIM
    if family in GNN_FAMILIES:
        return MolGNN(ATOM_DIM, hidden=params['hidden'], n_layers=params['n_layers'],
                      dropout=params['dropout'], conv_type=family,
                      heads=params.get('heads', 4), extra_dim=extra_dim).to(device)
    if family == 'Graphormer':
        from graphormer_true import TrueGraphormer
        return TrueGraphormer(ATOM_DIM, hidden=params['hidden'],
                              num_heads=params['num_heads'],
                              num_layers=params['num_layers'],
                              dropout=params['dropout'], extra_dim=extra_dim).to(device)
    if family == 'GraphGPS':
        from gps_model import GraphGPSModel
        return GraphGPSModel(ATOM_DIM, hidden=params['hidden'],
                             num_heads=params['num_heads'],
                             num_layers=params['num_layers'],
                             dropout=params['dropout'], extra_dim=extra_dim).to(device)
    if family == 'MVGNN':
        from mvgnn_model import MVGNNCross
        return MVGNNCross(hidden=params['hidden'], n_layers=params['n_layers'],
                          dropout=params['dropout'], extra_dim=extra_dim).to(device)
    raise ValueError(family)


def _forward(model, family, batch, device, use_extra):
    extra = batch.extra.to(device) if use_extra else None
    if family in GNN_FAMILIES:
        return model(batch.x, batch.edge_index, batch.batch, extra=extra)
    if family == 'Graphormer':
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                     batch.spd_flat, batch.edgepath_flat, extra=extra)
    if family == 'GraphGPS':
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                     batch.pe, extra=extra)
    raise ValueError(family)


def _predict_graph(model, family, loader, device, use_extra):
    import torch
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logit = _forward(model, family, batch, device, use_extra)
            probs.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


def _run_graph(family, params, data, predict_sets, extra_predict, device,
               max_epochs, patience, epoch_callback=None, init_state=None):
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader
    use_extra = data.extra is not None
    loaders = {s: DataLoader(data.graph_list(s), batch_size=BATCH, shuffle=(s == 'train'))
               for s in set(list(predict_sets) + ['train', 'val'])}
    model = _build_graph_model(family, params, data.extra_dim, device)
    if init_state is not None:
        # Stage-2 warm start: continue from the Stage-1 weights rather than
        # retraining from scratch, matching train_gnn_stage2.py in DILI_proxy_clean
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'],
                           weight_decay=params.get('weight_decay', 0.0))
    y_tr = data.labels('train')
    pos = max(int((y_tr == 1).sum()), 1)
    pw = torch.tensor([float((y_tr == 0).sum()) / pos], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for batch in loaders['train']:
            batch = batch.to(device)
            opt.zero_grad()
            loss = crit(_forward(model, family, batch, device, use_extra),
                        batch.y.view(-1))
            loss.backward()
            opt.step()
        va_prob, va_y = _predict_graph(model, family, loaders['val'], device, use_extra)
        auc = roc_auc_score(va_y, va_prob) if len(np.unique(va_y)) > 1 else 0.5
        if auc > best_auc:
            best_auc, best_state, no_improve = auc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        # lets an Optuna trial report progress and be pruned; the callback raises
        # to abort, so a pruned trial costs only the epochs it actually ran
        if epoch_callback is not None:
            epoch_callback(epoch, best_auc)

    model.load_state_dict(best_state)
    probs = {s: _predict_graph(model, family, loaders[s], device, use_extra)[0]
             for s in predict_sets}
    for name, graph_list in (extra_predict or {}).items():
        ldr = DataLoader(graph_list, batch_size=BATCH, shuffle=False)
        probs[name] = _predict_graph(model, family, ldr, device, use_extra)[0]
    return best_auc, probs, model


# ── MV-GNN (two graph views, paired loader) ──────────────────────────────────
def _predict_mvgnn(model, loader, device, use_extra):
    import torch
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for atom_b, bond_b in loader:
            atom_b, bond_b = atom_b.to(device), bond_b.to(device)
            extra = atom_b.extra.to(device) if use_extra else None
            la, lb = model(atom_b, bond_b, extra=extra)
            probs.append(((torch.sigmoid(la) + torch.sigmoid(lb)) / 2).cpu().numpy())
            ys.append(atom_b.y.view(-1).cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


def _run_mvgnn(params, data, predict_sets, extra_predict, device, max_epochs,
               patience, epoch_callback=None, init_state=None):
    import torch
    from mvgnn_model import MVGNNCross, make_paired_loader, mvgnn_loss
    use_extra = data.extra is not None
    loaders = {s: make_paired_loader(data.graph_list(s), data.graph_list(s, bond=True),
                                     BATCH, shuffle=(s == 'train'))
               for s in set(list(predict_sets) + ['train', 'val'])}
    model = MVGNNCross(hidden=params['hidden'], n_layers=params['n_layers'],
                       dropout=params['dropout'], extra_dim=data.extra_dim).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    opt = torch.optim.Adam(model.parameters(), lr=params['lr'],
                           weight_decay=params.get('wd', 0.0))
    y_tr = data.labels('train')
    pos = max(int((y_tr == 1).sum()), 1)
    pw = torch.tensor([float((y_tr == 0).sum()) / pos], device=device)

    best_auc, best_state, no_improve = -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for atom_b, bond_b in loaders['train']:
            atom_b, bond_b = atom_b.to(device), bond_b.to(device)
            opt.zero_grad()
            extra = atom_b.extra.to(device) if use_extra else None
            la, lb = model(atom_b, bond_b, extra=extra)
            loss, _, _ = mvgnn_loss(la, lb, atom_b.y.view(-1), pw, params['lam'])
            loss.backward()
            opt.step()
        va_prob, va_y = _predict_mvgnn(model, loaders['val'], device, use_extra)
        auc = roc_auc_score(va_y, va_prob) if len(np.unique(va_y)) > 1 else 0.5
        if auc > best_auc:
            best_auc, best_state, no_improve = auc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        if epoch_callback is not None:
            epoch_callback(epoch, best_auc)

    model.load_state_dict(best_state)
    probs = {s: _predict_mvgnn(model, loaders[s], device, use_extra)[0] for s in predict_sets}
    for name, (atom_list, bond_list) in (extra_predict or {}).items():
        ldr = make_paired_loader(atom_list, bond_list, BATCH, shuffle=False)
        probs[name] = _predict_mvgnn(model, ldr, device, use_extra)[0]
    return best_auc, probs, model


# ── public entry point ───────────────────────────────────────────────────────
def run(family, params, data, predict_sets=('val', 'test'), extra_predict=None,
        device='cpu', max_epochs=MAX_EPOCHS, patience=PATIENCE, epoch_callback=None,
        init_state=None, calibrate=True):
    """Trains one (family, hyperparameter) combination and predicts.

    Returns (val_auc, probs, model). `probs` maps each name in predict_sets, plus
    each key of extra_predict, to a probability vector.

    extra_predict carries out-of-split prediction targets (the DILI compounds for
    the assay models): {name: X_array or graph_list or (atom_list, bond_list)},
    matching what the family consumes.

    epoch_callback(epoch, best_val_auc_so_far) is called after each epoch of a
    graph model, so an Optuna trial can report progress and prune. Ignored by the
    classical families, which have no epochs.

    calibrate=False makes an SVM search skip Platt calibration and rank with
    decision_function() instead; ROC-AUC is unchanged by that, and it removes the
    dominant cost of the search. Leave it True for any run whose probabilities
    are used downstream.

    init_state warm-starts a graph model from a saved state_dict, which is how
    the DILI Stage-2 ("Proxy+DILI") fine-tune continues from Stage-1 instead of
    retraining from scratch. Ignored by the classical families, which have no
    continuable weights and are re-fitted independently.
    """
    np.random.seed(SEED)
    if family in ('RF', 'SVM', 'XGB'):
        return _run_classical(family, params, data, predict_sets, extra_predict,
                              calibrate=calibrate)
    import torch
    torch.manual_seed(SEED)
    if family == 'MVGNN':
        return _run_mvgnn(params, data, predict_sets, extra_predict, device,
                          max_epochs, patience, epoch_callback, init_state)
    return _run_graph(family, params, data, predict_sets, extra_predict, device,
                      max_epochs, patience, epoch_callback, init_state)
