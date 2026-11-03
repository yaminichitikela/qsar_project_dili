"""Asserts features.py reproduces the canonical project featurization exactly.

If this fails, the in vitro features would describe molecules differently from
the models that produced them, so every downstream number would be suspect.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import features as F


def load(path, name):
    """Imports a canonical script by path without running its main()."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    return mod


ref_split = load(f'{REB}/Summary_1_models/modeling/wahab_jobs/prepare_random_repeat_splits.py',
                 'ref_split')
ref_bond = load(f'{REB}/Summary_1_models/modeling/wahab_jobs/prepare_mvgnn_bonds.py', 'ref_bond')
ref_gt = load(f'{REB}/Summary_1_models/modeling/wahab_jobs/gt_model.py', 'ref_gt')

SMILES = ['CC(=O)Oc1ccccc1C(=O)O', 'CN1C=NC2=C1C(=O)N(C)C(=O)N2C', 'c1ccccc1',
          'CCO', 'C', '[Na+].[Cl-]', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
          'O=C(N)c1ccccc1O', 'C1CC1N2CCOCC2', 'CSc1ccc(cc1)C(=O)O']
fails = []


def check(name, cond, detail=''):
    print(f'   {name:46s} {"ok" if cond else "MISMATCH " + detail}')
    if not cond:
        fails.append(name)


print('Comparing features.py against the canonical scripts')
for smi in SMILES:
    cs_a, cs_b = F.canonicalize(smi), ref_split.canonicalize(smi)
    check(f'canonicalize   {smi[:22]:22s}', cs_a == cs_b)

    fp_a, fp_b = F.smiles_to_ecfp4(cs_a), ref_split.smiles_to_ecfp4(cs_b)
    check(f'ecfp4          {smi[:22]:22s}', np.array_equal(fp_a, fp_b))

    g_a, g_b = F.mol_to_graph(cs_a, 1), ref_split.mol_to_graph(cs_b, 1)
    same = (torch.equal(g_a.x, g_b.x)
            and torch.equal(g_a.edge_index, g_b.edge_index)
            and torch.equal(g_a.edge_attr, g_b.edge_attr)
            and torch.equal(g_a.y, g_b.y))
    check(f'atom graph     {smi[:22]:22s}', same)
    check(f'atom dim == 27 {smi[:22]:22s}', g_a.x.shape[1] == F.ATOM_DIM)

    b_a, b_b = F.atom_graph_to_bond_graph(g_a), ref_bond.atom_graph_to_bond_graph(g_b)
    same = (torch.equal(b_a.x, b_b.x)
            and torch.equal(b_a.edge_index, b_b.edge_index)
            and torch.equal(b_a.edge_attr, b_b.edge_attr))
    check(f'bond graph     {smi[:22]:22s}', same)

    pe_a, pe_b = F.compute_laplacian_pe(g_a), ref_gt.compute_laplacian_pe(g_b)
    check(f'laplacian PE   {smi[:22]:22s}', torch.allclose(pe_a, pe_b, atol=1e-6))

print('\n' + '=' * 60)
print('FEATURIZATION IDENTICAL' if not fails
      else f'{len(fails)} MISMATCHES:\n  - ' + '\n  - '.join(fails))
sys.exit(1 if fails else 0)
