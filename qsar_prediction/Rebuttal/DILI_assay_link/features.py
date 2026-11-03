"""Shared molecular featurization for the DILI-linking experiment.

Every function here is copied verbatim (same constants, same ordering, same
RDKit calls) from the canonical scripts already used project-wide:

  canonicalize, smiles_to_ecfp4, atom_feat, bond_feat, mol_to_graph
      <- Summary_1_models/modeling/wahab_jobs/prepare_random_repeat_splits.py
  atom_graph_to_bond_graph
      <- Summary_1_models/modeling/wahab_jobs/prepare_mvgnn_bonds.py
  compute_laplacian_pe
      <- Summary_1_models/modeling/wahab_jobs/gt_model.py

They live in one module here (rather than being re-pasted per script, as the
older folders do) so the assay side and the DILI side of this experiment cannot
silently drift apart in featurization -- the in vitro features would be
meaningless if the two sides encoded molecules differently.
test_features_match.py asserts byte-equality against the originals.

The only addition is smiles_to_inchikey, used for the leakage-safe exclusion of
DILI compounds from assay training (InChIKey, never SMILES -- matching on SMILES
is the bug prepare_proxy_data_clean.py was written to fix).
"""
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, inchi, rdchem
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj

RDLogger.DisableLog('rdApp.*')

ATOM_DIM = 27
BOND_DIM = 6
PE_DIM = 8


def canonicalize(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def smiles_to_inchikey(smi):
    """Standard InChIKey, used for cross-dataset identity matching."""
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return inchi.MolToInchiKey(mol) if mol else None
    except Exception:
        return None


def smiles_to_ecfp4(smi, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


ATOM_SYMBOLS = [6, 7, 8, 16, 9, 15, 17, 35, 53]  # C N O S F P Cl Br I

HYBRID_MAP = {
    rdchem.HybridizationType.SP: 0,
    rdchem.HybridizationType.SP2: 1,
    rdchem.HybridizationType.SP3: 2,
}

BOND_TYPES = [rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
              rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC]


def one_hot(val, choices):
    vec = [0] * len(choices)
    try:
        vec[choices.index(val)] = 1
    except ValueError:
        pass
    return vec


def atom_feat(atom):
    at_oh = one_hot(atom.GetAtomicNum(), ATOM_SYMBOLS) + [0]
    hyb = [0, 0, 0, 0]
    hyb[HYBRID_MAP.get(atom.GetHybridization(), 3)] = 1
    deg = min(atom.GetDegree(), 5)
    deg_oh = [0] * 6
    deg_oh[deg] = 1
    fc = atom.GetFormalCharge()
    fc_enc = [1 if fc < 0 else 0, 1 if fc == 0 else 0, 1 if fc > 0 else 0]
    extras = [
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        min(atom.GetTotalNumHs(), 4) / 4.0,
        atom.GetMass() / 100.0,
    ]
    return at_oh + deg_oh + fc_enc + hyb + extras  # 27-dim


def bond_feat(bond):
    bt = one_hot(bond.GetBondType(), BOND_TYPES)
    return bt + [int(bond.GetIsConjugated()), int(bond.IsInRing())]  # 6-dim


def mol_to_graph(smi, label):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    x = torch.tensor([atom_feat(a) for a in mol.GetAtoms()], dtype=torch.float)
    ei, ea = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_feat(bond)
        ei += [[i, j], [j, i]]
        ea += [bf, bf]
    if ei:
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(ea, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 6), dtype=torch.float)
    y = torch.tensor([label], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def atom_graph_to_bond_graph(data):
    """MV-GNN^cross line-graph view: each bond becomes a node, two bond-nodes
    are linked when they share an atom, and that atom's 27-dim feature is the
    edge feature."""
    x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y

    ei = edge_index.t().tolist()
    ea = edge_attr.tolist()

    seen, bond_list = {}, []
    for (i, j), bf in zip(ei, ea):
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen[key] = len(bond_list)
            bond_list.append((i, j, bf))

    if len(bond_list) == 0:
        return Data(x=torch.zeros((1, BOND_DIM), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, ATOM_DIM), dtype=torch.float), y=y)

    bx = torch.tensor([bf for _, _, bf in bond_list], dtype=torch.float)

    atom_to_bonds = {}
    for bi, (i, j, _) in enumerate(bond_list):
        atom_to_bonds.setdefault(i, []).append(bi)
        atom_to_bonds.setdefault(j, []).append(bi)

    edges, attrs = [], []
    for atom_idx, blist in atom_to_bonds.items():
        if len(blist) < 2:
            continue
        af = x[atom_idx].tolist()
        for a in range(len(blist)):
            for b in range(len(blist)):
                if a != b:
                    edges.append([blist[a], blist[b]])
                    attrs.append(af)

    if edges:
        bei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        bea = torch.tensor(attrs, dtype=torch.float)
    else:
        bei = torch.zeros((2, 0), dtype=torch.long)
        bea = torch.zeros((0, ATOM_DIM), dtype=torch.float)
    return Data(x=bx, edge_index=bei, edge_attr=bea, y=y)


def compute_laplacian_pe(data, k=PE_DIM):
    """Laplacian positional encoding for GraphGPS.

    NOTE ON SIGN: eigenvector signs are arbitrary (+v and -v are both valid), so
    the sign stored here carries no chemical meaning. The GPS paper handles this
    with random sign flipping during training; gps_model.py applies that at
    train time rather than baking a fixed arbitrary sign into the cached graphs.
    """
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
