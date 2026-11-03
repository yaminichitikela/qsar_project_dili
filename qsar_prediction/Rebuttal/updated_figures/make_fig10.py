"""Draw Figure 10 -- BRICS fragments enriched in exactly one cell line.

Reads the unique_fragments_{old,new}.csv written by unique_frags.py and draws
the summary-assay version in the layout of the published figure, with a
side-by-side statistics table for the old and new assays.
"""
import io
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))

# trivial fragments the published figure does not show
EXCLUDE = {'[7*]CC([7*])C'}

NAMES = {
    '[13*]C1CO1':              'Epoxide',
    '[7*]CC1=C(C)CCCC1(C)C':   '1,2,3,3-Tetramethylcyclohexene',
    '[7*]CC=O':                'Acetaldehyde',
    '[8*]C([8*])C(Cl)Cl':      '1,1-Dichloroethane',
    '[16*]c1ccc(Cl)cc1':       'Chlorobenzene',
    '[1*]C(=O)C=C':            'Acryloyl (vinyl ketone)',
    '[4*]CCCl':                '2-Chloroethyl',
    '[7*]CC([7*])C':           'Isopropyl',
}
ORDER = ['SK-N-SH', 'H-4-II-E', 'HEK293', 'CHO-K1', 'Jurkat']


def clean_frag(s):
    return re.sub(r'\[\d+\*\]', '[H]', s)


def draw(frag):
    mol = Chem.MolFromSmiles(clean_frag(frag))
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(620, 500)
    o = d.drawOptions()
    o.bondLineWidth = 4.5
    o.minFontSize = 30
    o.maxFontSize = 70
    o.padding = 0.14
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def main():
    new = pd.read_csv(f'{HERE}/unique_fragments_new.csv')
    old = pd.read_csv(f'{HERE}/unique_fragments_old.csv')
    new = new[~new.fragment.isin(EXCLUDE)].copy()

    new['rank'] = new.cell_line.map({c: i for i, c in enumerate(ORDER)})
    new = new.sort_values(['rank', 'padj']).reset_index(drop=True)

    nrow = 2
    ncol = int(np.ceil(len(new) / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 4.3 * nrow),
                             squeeze=False)
    for ax, (_, r) in zip(axes.flat, new.iterrows()):
        img = draw(r.fragment)
        if img is None:
            ax.axis('off')
            continue
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'{NAMES.get(r.fragment, r.fragment)}\n[{r.cell_line}]',
                     fontsize=17, fontweight='normal', pad=8, linespacing=1.45)
    for ax in axes.flat[len(new):]:
        ax.axis('off')
    fig.tight_layout()
    path = f'{HERE}/fig10_cellline_fragments_summary.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {path}')

    # side-by-side statistics
    o = old.set_index(['cell_line', 'fragment'])
    n = new.set_index(['cell_line', 'fragment'])
    keys = list(dict.fromkeys(list(o.index) + list(n.index)))
    rows = []
    for k in keys:
        if k[1] in EXCLUDE:
            continue
        rows.append({
            'cell_line': k[0], 'fragment': k[1],
            'name': NAMES.get(k[1], ''),
            'OR_old': o.odds_ratio.get(k, np.nan), 'padj_old': o.padj.get(k, np.nan),
            'OR_new': n.odds_ratio.get(k, np.nan), 'padj_new': n.padj.get(k, np.nan),
        })
    cmp = pd.DataFrame(rows)
    cmp.to_csv(f'{HERE}/fig10_old_vs_new.csv', index=False)
    print()
    print(cmp.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    main()
