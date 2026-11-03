"""Assembles the SubgraphX results into the composite panels the manuscript uses.

Reads results/xai/subgraphx_alerts.csv, which carries the atom mask for each
explained compound, and redraws each molecule into a 3-column grid per model. No
search is re-run, so the panels always match the CSV.

Usage: python make_xai_panels.py [n_compounds|all]

n_compounds is how many of the top-ranked compounds to draw (default 6, laid out
3 per row). "all" draws every compound in the CSV -- 118 for Graphormer, 101 for
GraphGPS and 85 for the consensus -- which is an SI-sized figure, not a main-text
one. Every compound listed in the CSV already has its mask stored, so enlarging
the grid never re-runs the Shapley search.

Outputs (../figures/): subgraphx_graphormer.png, subgraphx_graphgps.png,
subgraphx_consensus.png (and .pdf for each)
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

HERE = os.path.dirname(os.path.abspath(__file__))
XAI = os.path.join(HERE, 'results', 'xai')
FIGS = os.path.join(os.path.dirname(HERE), 'figures')
INK, MUTED = '#0b0b0b', '#52514e'
RED = (0.90, 0.20, 0.20)


def png_for(row, path, size=(430, 330)):
    mol = Chem.MolFromSmiles(row.smiles)
    if mol is None:
        return False
    mask = [int(v) for v in str(row['mask']).split()]
    atoms = [i for i, v in enumerate(mask) if v and i < mol.GetNumAtoms()]
    bonds, bcol = [], {}
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms:
            bonds.append(b.GetIdx()); bcol[b.GetIdx()] = RED
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().clearBackground = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        d, mol, highlightAtoms=atoms, highlightAtomColors={i: RED for i in atoms},
        highlightBonds=bonds, highlightBondColors=bcol)
    d.FinishDrawing()
    open(path, 'wb').write(d.GetDrawingText())
    return True


def panel(df, model, out, n=6, ncol=3):
    sub = df.sort_values('rank') if model == 'Consensus' else \
        df[df.model == model].sort_values('rank')
    sub = sub.head(n)
    nrow = max(1, -(-len(sub) // ncol))          # ceil, so the last row may be partial
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.6, 3.0 * nrow), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
        tmp = os.path.join(XAI, f'{model.lower()}_tp{int(r["rank"])}.png')
        ok = png_for(r, tmp)
        ax.axis('off')
        if not ok:
            continue
        ax.imshow(mpimg.imread(tmp))
        frag = r.fragment if isinstance(r.fragment, str) else '--'
        label = {'Consensus': 'Consensus-GT', 'Consensus-GNN': 'Consensus-GNN'}.get(model)
        title = f'{label}  p = {r.prob:.3f}' if label else f'p = {r.prob:.3f}'
        ax.set_title(title, fontsize=10, color=INK, pad=2)
        ax.text(0.5, -0.04, f'{int(r.n_highlighted)} of {int(r.n_atoms)} atoms  ·  {frag}',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=8.4, color=MUTED)
    for ax in axes.ravel()[len(sub):]:
        ax.axis('off')
    # top/bottom are per-figure fractions, so they have to shrink as rows are
    # added or the titles eat into the first row's structure.
    fig.subplots_adjust(left=.01, right=.99, top=1 - .05 / nrow * 2,
                        bottom=.06 / nrow * 2, wspace=.02, hspace=.22)
    fig.savefig(out + '.png', dpi=300, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}.png / .pdf  ({len(sub)} compounds)')


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else '6'
    n = 10 ** 9 if arg == 'all' else int(arg)
    f = os.path.join(XAI, 'subgraphx_alerts.csv')
    if not os.path.exists(f):
        raise SystemExit(f'missing {f} -- run run_subgraphx.py first')
    df = pd.read_csv(f)
    for model, stem in (('Graphormer', 'subgraphx_graphormer'),
                        ('GraphGPS', 'subgraphx_graphgps')):
        panel(df, model, os.path.join(FIGS, stem), n=n)
    agree = os.path.join(XAI, 'subgraphx_agreement.csv')
    if os.path.exists(agree):
        panel(pd.read_csv(agree), 'Consensus',
              os.path.join(FIGS, 'subgraphx_consensus'), n=n)
    cgnn = os.path.join(XAI, 'subgraphx_consensus_gnn.csv')
    if os.path.exists(cgnn):
        panel(pd.read_csv(cgnn), 'Consensus-GNN',
              os.path.join(FIGS, 'subgraphx_consensus_gnn'), n=n)


if __name__ == '__main__':
    main()
