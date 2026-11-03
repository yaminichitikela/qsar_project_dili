"""Three-panel manuscript figure: DILI-positive rate by in vitro pattern,
under measured-only, measured+predicted, and predicted-only labelling.

Regenerates the right-hand panel of the three overlap_analysis.ipynb figures
side by side. Run from the DILI_assay_link/ folder.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

def to_inchikey(smi):
    mol = Chem.MolFromSmiles(str(smi))
    return Chem.MolToInchiKey(mol) if mol else None

ASSAYS = {'mmp': 'inputs/AID_720637_cleaned.csv',
          'caspase': 'inputs/AID_1347034_cleaned.csv'}
LBL = {(0, 0): 'neither\nactive', (1, 0): 'caspase\nonly',
       (0, 1): 'MMP\nonly', (1, 1): 'both\nactive'}
BAR, EDGE = '#5b82ad', '#2f4e70'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif', 'axes.linewidth': 0.8,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
})

def wilson(k, n, z=1.96):
    """95% Wilson interval - honest at the n=1/n=2 cells of the measured panel."""
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z**2 / n
    c, h = (p + z**2 / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0, c - h), min(1, c + h)

# ---- data ------------------------------------------------------------------
held = pd.read_csv('inputs/Held-223_deduped.csv', usecols=['InChIKey', 'TOXICITY'])
ncv = pd.read_csv('inputs/NCV-888_deduped.csv', usecols=['InChIKey', 'TOXICITY'])
held['set'], ncv['set'] = 'Held-223', 'NCV-888'
gold = (pd.concat([held, ncv]).dropna(subset=['InChIKey']).drop_duplicates('InChIKey')
          .rename(columns={'TOXICITY': 'dili'}).set_index('InChIKey'))

def load_assay(path):
    d = pd.read_csv(path, usecols=['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME'])
    d = d[d.PUBCHEM_ACTIVITY_OUTCOME.isin(['Active', 'Inactive'])]
    d['ikey'] = d.PUBCHEM_EXT_DATASOURCE_SMILES.map(to_inchikey)
    d['y'] = (d.PUBCHEM_ACTIVITY_OUTCOME == 'Active').astype(int)
    m = d.dropna(subset=['ikey']).groupby('ikey').y.mean()
    return (m[m != 0.5] > 0.5).astype(int)

lab = {a: load_assay(p) for a, p in ASSAYS.items()}
prob = pd.read_csv('results/invitro_features.csv',
                   usecols=['InChIKey', 'mmp_prob', 'caspase_prob']).set_index('InChIKey')

thresh = {}
for a in lab:
    m = pd.concat([lab[a].rename('y'), prob[f'{a}_prob']], axis=1, join='inner')
    m = m[m.index.isin(gold.index)]
    fpr, tpr, cuts = roc_curve(m.y, m[f'{a}_prob'])
    thresh[a] = float(cuts[np.argmax(tpr - fpr)])

meas = gold.copy()
for a in lab:
    meas[a] = lab[a].reindex(meas.index)
tri = meas.dropna(subset=['caspase', 'mmp']).astype({'caspase': int, 'mmp': int})

hyb = gold.join(prob, how='inner')
pred = hyb.copy()
for a in lab:
    measured = lab[a].reindex(hyb.index)
    hyb[a] = measured.fillna((hyb[f'{a}_prob'] >= thresh[a]).astype(int)).astype(int)
    pred[a] = (pred[f'{a}_prob'] >= thresh[a]).astype(int)

PANELS = [('A', tri, 'Measured only'),
          ('B', hyb, 'Measured + predicted'),
          ('C', pred, 'Predicted only')]

# ---- figure ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9), sharey=True)
for ax, (tag, df, title) in zip(axes, PANELS):
    g = df.groupby(['caspase', 'mmp']).dili.agg(n='size', pos='sum')
    g = g.reindex(list(LBL), fill_value=0)
    rate = 100 * g.pos / g.n.where(g.n > 0, 1)
    lo_hi = np.array([wilson(k, n) for k, n in zip(g.pos, g.n)]) * 100
    err = np.abs(lo_hi.T - rate.values)
    base = 100 * df.dili.mean()
    x = np.arange(4)

    ax.bar(x, rate, width=0.6, color=BAR, edgecolor=EDGE, linewidth=0.8, zorder=3)
    ax.errorbar(x, rate, yerr=err, fmt='none', ecolor='black', elinewidth=0.8,
                capsize=2.5, capthick=0.8, zorder=4)
    ax.axhline(base, color='black', ls=(0, (5, 4)), lw=0.8, zorder=2)
    ax.text(3.88, base + 1.5, f'{base:.1f}%', ha='right', va='bottom', fontsize=7.5)

    for xi, (r, hi) in enumerate(zip(rate, lo_hi[:, 1])):
        ax.text(xi, hi + 2.5, f'{r:.0f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n($n$={n})' for t, n in zip(LBL.values(), g.n)],
                       fontsize=8.5)
    ax.set_xlim(-0.65, 3.9)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(f'({tag}) {title}, $n$={len(df):,}', fontsize=9.5, pad=7)
    ax.tick_params(labelsize=8.5)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

axes[0].set_ylabel('DILI-positive rate (%)', fontsize=9)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'results/fig_dili_rate_by_invitro_pattern.{ext}', dpi=300, bbox_inches='tight')

# ---- clean validation of the predicted calls -------------------------------
# The Youden-J cutoff is fitted on the same gold compounds whose agreement is
# then reported, so that agreement is optimistic. Two leak-free estimates are
# printed instead: a cutoff-free ROC-AUC, and agreement under a 5-fold
# cross-validated cutoff (fit on 4 folds, scored on the held-out fold). A
# cutoff taken from the non-gold assay compounds is NOT used -- it transfers
# badly because the whole-assay active rate differs from the gold subset's.
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
print()
for a in lab:
    m = pd.concat([lab[a].rename('y'), prob[f'{a}_prob'].rename('p')],
                  axis=1, join='inner').dropna()
    m = m[m.index.isin(gold.index)]
    cv = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(m, m.y):
        fpr, tpr, cuts = roc_curve(m.y.iloc[tr], m.p.iloc[tr])
        t = float(cuts[np.argmax(tpr - fpr)])
        cv.append(((m.p.iloc[te] >= t).astype(int) == m.y.iloc[te]).mean())
    print(f'{a:8s} n={len(m):4d}  ROC-AUC={roc_auc_score(m.y, m.p):.3f}  '
          f'CV-cutoff agreement={100*np.mean(cv):.1f}% +/- {100*np.std(cv):.1f}%  '
          f'(in-sample cutoff would give {100*((m.p >= thresh[a]).astype(int) == m.y).mean():.1f}%)')
print()

for tag, df, title in PANELS:
    g = df.groupby(['caspase', 'mmp']).dili.agg(n='size', pos='sum').reindex(list(LBL), fill_value=0)
    r = 100 * g.pos / g.n.where(g.n > 0, 1)
    print(f'({tag}) {title:22s} base={100*df.dili.mean():.1f}%  ' +
          '  '.join(f'{LBL[i].replace(chr(10)," ")}: {r[i]:.0f}% (n={g.n[i]})' for i in LBL))
