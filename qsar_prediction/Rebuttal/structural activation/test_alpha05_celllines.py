"""Does any cell-line BRICS result depend on the relaxed FDR threshold?

Reviewer comment 3 objects that the per-cell-line enrichment uses BH-FDR
alpha = 0.10 on foregrounds of only 10-19 active compounds. This re-runs that
analysis unchanged except for the threshold, and reports every fragment that
falls in the 0.05 < p_adj <= 0.10 band -- i.e. the ones that exist only because
the threshold was relaxed. If that band is empty, no reported result depends on
alpha = 0.10 and the caption can state alpha = 0.05 without changing a number.

Method is transcribed from pairwise_brics_comparison.ipynb (Analysis C):
per cell line, foreground = compounds active in that line, background = the
other compounds in the at-least-one-active set; one-sided Fisher's exact test
per BRICS fragment, Benjamini-Hochberg correction, min_count = 2, cell lines
with fewer than 10 actives skipped.

Usage: python test_alpha05_celllines.py
"""
import os
import re
import warnings

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CSV = os.path.join(ROOT, 'substructure_analysis', 'Casepase-activation', 'results',
                   'all_cell_line_comparisions', 'caspase_atleast_one_active.csv')
# the cross-cell-line file holds CIDs only; SMILES come from the cleaned assay
# tables, exactly as load_cid_smiles() does in the notebook
CLEANED = os.path.join(ROOT, 'qsar_prediction', 'cleaned_datasets')

AID_COLS = ['AID_1346980', 'AID_655', 'AID_656', 'AID_657', 'AID_658', 'AID_659',
            'AID_660', 'AID_661', 'AID_663', 'AID_664', 'AID_665', 'AID_666',
            'AID_667', 'combined_hepg2_casepase.csv']
CELL_NAMES = {'AID_1346980': 'CHO-K1', 'AID_655': 'Jurkat', 'AID_656': 'HUV-EC-C',
              'AID_657': 'SH-SY5Y', 'AID_658': 'BJ', 'AID_659': 'MRC-5',
              'AID_660': 'Mesangial', 'AID_661': 'SK-N-SH', 'AID_663': 'H-4-II-E',
              'AID_664': 'HEK293', 'AID_665': 'N2a', 'AID_666': 'NIH 3T3',
              'AID_667': 'Renal PT', 'combined_hepg2_casepase.csv': 'HepG2'}
MIN_ACTIVE, MIN_COUNT = 10, 2


def brics_frags(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    return frozenset(BRICS.BRICSDecompose(mol)) if mol else frozenset()


def enrich(fg, bg, min_count=MIN_COUNT):
    """One-sided Fisher per fragment + BH correction. Threshold applied later."""
    na, nb = len(fg), len(bg)
    rows = []
    for frag in set().union(*fg, *bg) if (fg and bg) else []:
        a = sum(1 for s in fg if frag in s)
        if a < min_count:
            continue
        c = sum(1 for s in bg if frag in s)
        odds, p = fisher_exact([[a, na - a], [c, nb - c]], alternative='greater')
        rows.append({'fragment': re.sub(r'\[\d+\*\]', '[H]', frag),
                     'n_a': a, 'N_a': na, 'n_b': c, 'N_b': nb,
                     'odds_ratio': odds, 'pval': p})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d['padj'] = multipletests(d.pval, method='fdr_bh')[1]
    return d.sort_values('padj')


def main():
    cross = pd.read_csv(CSV, low_memory=False)
    lut = {}
    for f in sorted(os.listdir(CLEANED)):
        if not f.endswith('.csv'):
            continue
        try:
            d = pd.read_csv(os.path.join(CLEANED, f), low_memory=False,
                            usecols=['PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES'])
        except (ValueError, KeyError):
            continue
        d = d.dropna(subset=['PUBCHEM_CID', 'PUBCHEM_EXT_DATASOURCE_SMILES'])
        for cid, smi in zip(d.PUBCHEM_CID.astype(int), d.PUBCHEM_EXT_DATASOURCE_SMILES):
            lut.setdefault(cid, smi)
    cross['smiles'] = cross.PUBCHEM_CID.astype(int).map(lut)
    miss = cross.smiles.isna().sum()
    print(f'CID->SMILES lookup: {len(lut):,} entries; {miss} of {len(cross)} CIDs unmatched')
    cross = cross.dropna(subset=['smiles']).copy()
    cross['frags'] = cross.smiles.apply(brics_frags)
    cross = cross[cross.frags.map(len) > 0].reset_index(drop=True)
    print(f'compounds with parsable structures: {len(cross)}\n')

    band, total5, total10 = [], 0, 0
    for col in AID_COLS:
        name = CELL_NAMES[col]
        mask = cross[col] == 'Active'
        if mask.sum() < MIN_ACTIVE:
            continue
        d = enrich(list(cross[mask].frags), list(cross[~mask].frags))
        if d.empty:
            print(f'{name:12s} N_a={mask.sum():3d}   no fragment reached min_count')
            continue
        s5, s10 = d[d.padj <= 0.05], d[d.padj <= 0.10]
        total5 += len(s5)
        total10 += len(s10)
        print(f'{name:12s} N_a={mask.sum():3d}   significant: '
              f'{len(s5)} at alpha=0.05, {len(s10)} at alpha=0.10')
        for _, r in s10.iterrows():
            flag = '' if r.padj <= 0.05 else '   <-- ONLY at alpha=0.10'
            print(f'                 {r.fragment:28s} OR={r.odds_ratio:>7.1f} '
                  f'padj={r.padj:.4f}  {r.n_a}/{r.N_a}{flag}')
            if r.padj > 0.05:
                band.append({'cell_line': name, **r.to_dict()})

    print('\n' + '=' * 70)
    print(f'TOTAL significant: {total5} at alpha=0.05   |   {total10} at alpha=0.10')
    if band:
        print(f'\n{len(band)} fragment(s) exist ONLY because of the relaxed threshold:')
        print(pd.DataFrame(band)[['cell_line', 'fragment', 'odds_ratio', 'padj']]
              .to_string(index=False))
    else:
        print('\nNo fragment falls in 0.05 < p_adj <= 0.10.')
        print('Every reported result also satisfies alpha = 0.05.')
    pd.DataFrame(band).to_csv(f'{HERE}/alpha_band_005_010.csv', index=False)


if __name__ == '__main__':
    main()
