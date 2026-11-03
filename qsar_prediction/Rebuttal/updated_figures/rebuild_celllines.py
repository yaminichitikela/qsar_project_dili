"""Rebuild the cross-cell-line figures on the summary assays.

The published Figures 8-10 were built on the raw assays that were removed from
Table 1 and Table 3: HepG2 came from combined_hepg2_casepase.csv (AID 654 +
the raw AID 1346978, 241 actives) and CHO-K1 from the raw AID 1346980 (115
actives). This rebuilds them on the summary assays that the rest of the paper
uses -- HepG2 from AID 1347034 (303 actives) and CHO-K1 from AID 1347037 (171
actives) -- and reports the old and new numbers side by side.

Everything else is transcribed unchanged from pairwise_brics_comparison.ipynb
and split_top10_figure.py: the same BRICS decomposition, the same one-sided
Fisher test with Benjamini-Hochberg at alpha = 0.10, the same minimum of ten
actives per cell line, the same MCS highlighting and Morgan(2, 2048)
fingerprints. Only the HepG2 and CHO-K1 sources change.

Usage: python rebuild_celllines.py
"""
import io
import os
import re
import string
import warnings
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import BRICS, rdFingerprintGenerator, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
PROJ = '/Users/yamini/Desktop/projects/qsar casepase/'
CLEAN = PROJ + 'qsar_prediction/cleaned_datasets/'
REB = PROJ + 'qsar_prediction/Rebuttal/'

NCATS = {'AID_655': 'Jurkat', 'AID_656': 'HUV-EC-C', 'AID_657': 'SH-SY5Y',
         'AID_658': 'BJ', 'AID_659': 'MRC-5', 'AID_660': 'Mesangial',
         'AID_661': 'SK-N-SH', 'AID_663': 'H-4-II-E', 'AID_664': 'HEK293',
         'AID_665': 'N2a', 'AID_666': 'NIH 3T3', 'AID_667': 'Renal PT'}

# the two sources that change between the published figures and this rebuild
SOURCES = {
    'old': {'HepG2':  CLEAN + 'combined_hepg2_casepase.csv',
            'CHO-K1': CLEAN + 'AID_1346980_cleaned.csv'},
    'new': {'HepG2':  REB + 'summary_2_1437034/cleaning/AID_1347034_cleaned.csv',
            'CHO-K1': REB + 'summary_1347037/cleaning/AID_1347037_cleaned.csv'},
}

MIN_ACTIVE = 10
FDR_ALPHA = 0.10
MIN_COUNT = 2

TOP10_CIDS = [11057, 5288209, 2724411, 62770, 457193,
              108005, 443939, 2723598, 19042, 5284469]
PAIRS = {2724411: (11057, '#AED6F1'), 11057: (2724411, '#AED6F1'),
         62770: (443939, '#A9DFBF'), 443939: (62770, '#A9DFBF')}
MOL_SIZE = (900, 720)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ── loading ──────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path, usecols=lambda c: c in (
        'PUBCHEM_CID', 'PUBCHEM_ACTIVITY_OUTCOME',
        'PUBCHEM_EXT_DATASOURCE_SMILES'), low_memory=False)
    df = df.dropna(subset=['PUBCHEM_CID']).drop_duplicates('PUBCHEM_CID')
    df['PUBCHEM_CID'] = df['PUBCHEM_CID'].astype(int)
    return df


def build(variant):
    """Active-CID set per cell line, plus a CID -> SMILES map."""
    lines, smiles = {}, {}

    def absorb(name, path):
        d = load(path)
        lines[name] = set(d.loc[d.PUBCHEM_ACTIVITY_OUTCOME == 'Active', 'PUBCHEM_CID'])
        for cid, smi in zip(d.PUBCHEM_CID, d.PUBCHEM_EXT_DATASOURCE_SMILES):
            if pd.notna(smi):
                smiles.setdefault(cid, smi)

    for name, path in SOURCES[variant].items():
        absorb(name, path)
    for col, name in NCATS.items():
        absorb(name, CLEAN + f'{col}_cleaned.csv')
    return lines, smiles


# ── BRICS + Fisher, transcribed from pairwise_brics_comparison.ipynb ─────────
def brics_frags(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return frozenset(BRICS.BRICSDecompose(mol)) if mol else frozenset()


def clean_frag(frag_smi):
    return re.sub(r'\[\d+\*\]', '[H]', frag_smi)


def fisher_enrichment(fg, bg, label_a='A', label_b='other',
                      min_count=MIN_COUNT, fdr_alpha=FDR_ALPHA):
    na, nb = len(fg), len(bg)
    rows = []
    for frag in set().union(*fg, *bg) if (fg and bg) else set():
        a = sum(1 for fs in fg if frag in fs)
        if a < min_count:
            continue
        c = sum(1 for fs in bg if frag in fs)
        odds, pval = fisher_exact([[a, na - a], [c, nb - c]], alternative='greater')
        rows.append({'fragment': frag, f'n_{label_a}': a,
                     f'pct_{label_a}': 100 * a / na, f'n_{label_b}': c,
                     f'pct_{label_b}': 100 * c / nb,
                     'odds_ratio': odds, 'pvalue': pval})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['padj'] = multipletests(df['pvalue'], method='fdr_bh')[1]
    return df[df.padj < fdr_alpha].sort_values('padj').reset_index(drop=True)


def enrich_all(lines, smiles, tag):
    """Per-cell-line enrichment over the compounds active in >= 1 cell line."""
    counts = Counter()
    for actives in lines.values():
        counts.update(actives)
    cids = sorted(counts)

    frags = {}
    for cid in cids:
        smi = smiles.get(cid)
        fs = brics_frags(smi) if isinstance(smi, str) else frozenset()
        if fs:
            frags[cid] = fs
    valid = [c for c in cids if c in frags]
    print(f'  [{tag}] active in >=1 cell line: {len(cids)}  '
          f'with BRICS fragments: {len(valid)}')

    results, summary = {}, []
    for name in ['HepG2', 'CHO-K1'] + list(NCATS.values()):
        act = [c for c in valid if c in lines[name]]
        rest = [c for c in valid if c not in lines[name]]
        if len(act) < MIN_ACTIVE:
            summary.append({'cell_line': name, 'n_active': len(act),
                            'n_significant': None, 'note': 'skipped (<10 active)'})
            print(f'    {name:12s} active={len(act):4d}  skipped')
            continue
        res = fisher_enrichment([frags[c] for c in act],
                                [frags[c] for c in rest], label_a=name)
        results[name] = res
        summary.append({'cell_line': name, 'n_active': len(act),
                        'n_significant': len(res), 'note': ''})
        print(f'    {name:12s} active={len(act):4d}  enriched={len(res)}')
    return counts, frags, results, pd.DataFrame(summary)


# ── figure 10: enriched fragments per cell line ──────────────────────────────
def figure_fragments(results, path):
    items = [(name, row) for name, res in results.items()
             for _, row in res.iterrows()]
    if not items:
        print('  no significant fragments to draw')
        return None
    ncol = min(3, len(items))
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow),
                             squeeze=False)
    for ax, (name, row) in zip(axes.flat, items):
        mol = Chem.MolFromSmiles(clean_frag(row['fragment']))
        if mol is None:
            ax.axis('off')
            continue
        d = rdMolDraw2D.MolDraw2DCairo(600, 500)
        d.drawOptions().bondLineWidth = 3.0
        d.DrawMolecule(mol)
        d.FinishDrawing()
        ax.imshow(Image.open(io.BytesIO(d.GetDrawingText())))
        ax.axis('off')
        orv = row['odds_ratio']
        ors = '∞' if np.isinf(orv) else f'{orv:.1f}'
        ax.set_title(f'{row["fragment"]}\n[{name}]  OR={ors}  '
                     f'$p_{{adj}}$={row["padj"]:.3f}',
                     fontsize=11, fontweight='bold', pad=6, linespacing=1.4)
    for ax in axes.flat[len(items):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ── figures 8 and 9, transcribed from split_top10_figure.py ──────────────────
def mcs_mol(mols, a, b):
    res = rdFMCS.FindMCS([mols[a], mols[b]], timeout=30,
                         ringMatchesRingOnly=True, completeRingsOnly=True)
    return Chem.MolFromSmarts(res.smartsString)


def mol_to_pil(mol, highlight_atoms=None, highlight_color=None):
    d = rdMolDraw2D.MolDraw2DCairo(*MOL_SIZE)
    o = d.drawOptions()
    o.addStereoAnnotation = False
    o.bondLineWidth = 3.0
    o.padding = 0.08
    o.minFontSize = 10
    o.maxFontSize = 60
    if highlight_atoms:
        c = tuple(int(highlight_color.lstrip('#')[i:i + 2], 16) / 255 for i in (0, 2, 4))
        ac = {a: c for a in highlight_atoms}
        bl, bc = [], {}
        for bond in mol.GetBonds():
            if (bond.GetBeginAtomIdx() in highlight_atoms
                    and bond.GetEndAtomIdx() in highlight_atoms):
                bl.append(bond.GetIdx())
                bc[bond.GetIdx()] = c
        d.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=ac,
                       highlightBonds=bl, highlightBondColors=bc)
    else:
        d.DrawMolecule(mol)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def figure_structures(order, mols, nact, mcs_A, mcs_B, path):
    fig, axes = plt.subplots(4, 3, figsize=(17, 21))
    for i, cid in enumerate(order):
        ax = axes.flat[i]
        mol = mols[cid]
        hl, col = [], None
        if cid in PAIRS:
            partner, col = PAIRS[cid]
            ref = mcs_A if frozenset([cid, partner]) == frozenset([2724411, 11057]) else mcs_B
            hl = list(mol.GetSubstructMatch(ref))
        ax.imshow(mol_to_pil(mol, hl or None, col))
        ax.axis('off')
        ax.text(0.01, 0.99, f'({string.ascii_lowercase[i]})', transform=ax.transAxes,
                fontsize=19, fontweight='bold', va='top', ha='left')
        tag = ('  [Pair A]' if cid in (2724411, 11057)
               else '  [Pair B]' if cid in (62770, 443939) else '')
        ax.set_title(f'#{i + 1}  CID {cid}{tag}\n'
                     f'Active: {nact[cid]} / 14 cell lines',
                     fontsize=16, fontweight='bold', pad=6, linespacing=1.4)
    handles = [mpatches.Patch(facecolor='#AED6F1', edgecolor='#2980B9',
                              label='Pair A shared scaffold (CID 2724411 & 11057)'),
               mpatches.Patch(facecolor='#A9DFBF', edgecolor='#1E8449',
                              label='Pair B shared scaffold (CID 62770 & 443939)')]
    for ax in axes.flat[len(order):]:
        ax.axis('off')
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=15,
               frameon=True, framealpha=0.95, edgecolor='#AAAAAA',
               bbox_to_anchor=(0.5, 0.012))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def figure_heatmap(order, mols, path):
    fps = [gen.GetFingerprint(mols[c]) for c in order]
    sim = np.vstack([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(sim, cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(order, fontsize=10)
    ax.set_xlabel('PubChem CID', fontsize=11)
    ax.set_ylabel('PubChem CID', fontsize=11)
    for a, b, lab in [(2724411, 11057, 'Pair A'), (62770, 443939, 'Pair B')]:
        if a in order and b in order:
            ax.text(order.index(b), order.index(a), lab, ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white')
    plt.colorbar(im, ax=ax, label='Tanimoto similarity', shrink=0.85, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    off = sim[~np.eye(len(order), dtype=bool)]
    print(f'  off-diagonal Tanimoto: median {np.median(off):.3f}, max {off.max():.3f}')
    return path


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    store = {}
    for variant in ('old', 'new'):
        print(f'\n=== {variant} assays ===')
        lines, smiles = build(variant)
        for name in ['HepG2', 'CHO-K1']:
            print(f'  {name:8s} actives: {len(lines[name])}')
        counts, frags, results, summary = enrich_all(lines, smiles, variant)
        store[variant] = dict(lines=lines, smiles=smiles, counts=counts,
                              results=results, summary=summary)

    # side-by-side comparison of the per-cell-line counts
    cmp = store['old']['summary'].merge(
        store['new']['summary'], on='cell_line', suffixes=('_old', '_new'))
    cmp.to_csv(f'{OUT}/celllines_old_vs_new.csv', index=False)
    print('\n=== per-cell-line comparison ===')
    print(cmp.to_string(index=False))

    for variant in ('old', 'new'):
        for name, res in store[variant]['results'].items():
            if not res.empty:
                res.to_csv(f'{OUT}/fragments_{variant}_{name.replace("/", "-")}.csv',
                           index=False)

    # figure 10 on the new assays
    print('\n=== figure 10 ===')
    p = figure_fragments(store['new']['results'],
                         f'{OUT}/fig10_cellline_fragments_summary.png')
    print(f'  wrote {p}')

    # figures 8 and 9 on the new assays, same ten compounds
    print('\n=== figures 8 and 9 ===')
    nact = {c: store['new']['counts'].get(c, 0) for c in TOP10_CIDS}
    order = sorted(TOP10_CIDS, key=lambda c: -nact[c])
    smiles = store['new']['smiles']
    old_smiles = store['old']['smiles']
    mols = {}
    for c in TOP10_CIDS:
        smi = smiles.get(c) or old_smiles.get(c)
        mols[c] = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    missing = [c for c, m in mols.items() if m is None]
    if missing:
        print(f'  WARNING: no structure for {missing}')
        order = [c for c in order if mols[c] is not None]
    for c in order:
        print(f'  CID {c:>9}: {nact[c]} / 14 '
              f'(was {store["old"]["counts"].get(c, 0)})')
    a = figure_structures(order, mols, nact,
                          mcs_mol(mols, 2724411, 11057),
                          mcs_mol(mols, 62770, 443939),
                          f'{OUT}/fig8_top10_structures_summary.png')
    print(f'  wrote {a}')
    b = figure_heatmap(order, mols, f'{OUT}/fig9_tanimoto_summary.png')
    print(f'  wrote {b}')


if __name__ == '__main__':
    main()
