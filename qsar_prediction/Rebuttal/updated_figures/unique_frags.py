"""Fragments significant in exactly one cell line -- the view published as Figure 10."""
import glob, os, re
import pandas as pd

for variant in ('old', 'new'):
    rows = []
    for f in sorted(glob.glob(f'fragments_{variant}_*.csv')):
        cell = re.sub(rf'^fragments_{variant}_|\.csv$', '', os.path.basename(f))
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            rows.append({'cell_line': cell, 'fragment': r['fragment'],
                         'odds_ratio': r['odds_ratio'], 'padj': r['padj']})
    df = pd.DataFrame(rows)
    n_lines = df.groupby('fragment')['cell_line'].nunique()
    uniq = df[df.fragment.map(n_lines) == 1].sort_values(['cell_line', 'padj'])
    uniq.to_csv(f'unique_fragments_{variant}.csv', index=False)
    print(f'=== {variant}: {len(df)} significant, {len(uniq)} unique to one cell line ===')
    for _, r in uniq.iterrows():
        orv = 'inf' if r.odds_ratio == float('inf') else f'{r.odds_ratio:.1f}'
        star = '*' if r.padj < 0.05 else ' '
        print(f'  {r.cell_line:10s} {r.fragment:28s} OR={orv:>6}  padj={r.padj:.4f} {star}')
    print()
