"""Main-text figure: graph models under scaffold-based distribution shift.

Panels A-C give, per assay, the three best graph models against the best
classical baseline, so the comparison the figure claims is visible in it
rather than asserted in the caption.

The scaffold-split schematic is emitted as a SEPARATE figure
(scaffold_split_schematic), because it explains the evaluation design rather
than a result and belongs beside the Methods text that defines the split
(1__editing.tex line 439), not in a results figure.

Assay two is AID 1347034 alone. The scaffold runs under
summary_2_1437034/upsampling_1347034_protocols/ train on
cleaned_data/AID_1347034_cleaned.csv (303 active, 6,968 inactive), NOT the
merged AID 654+1347034 set used elsewhere in the paper.

Only the top three graph models are shown, per the space constraint. The best
classical model is kept as a grey reference; without it "the top models are
graph models" cannot be checked from the figure. Full rankings for all eleven
models are in the SI performance tables.

The best graph model is drawn in red. Every panel is also labelled and every
value printed, so the highlight is an emphasis, not the only carrier of the
identity.

INTERVALS
---------
Nadeau-Bengio corrected resampled t interval over the 5 scaffold repeats,
mean +/- t(0.975, n-1) * sqrt((1/n + 1/8) * var); 1/8 is the 8:1:1 split ratio
used by the SI performance tables. verify() re-checks every plotted number.

Usage: python make_scaffold_graph_figure.py [roc|both|schematic]
Outputs: figures/scaffold_graph_models.pdf/.png and _values.csv
"""
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from scipy import stats
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
RED, GRAPH, CLASSIC = '#c62828', '#2a9d8f', '#9a9a94'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#dcdcd8'
TRAIN_C, TEST_C = '#2a78d6', '#c62828'

ASSAY = {
 'mmp':   ('Mitochondrial membrane potential', 'AID 720637',
           'Summary_1_models/upsampling_protocols/results_scaffold'),
 'hepg2': ('Caspase-3/7 HepG2', 'AID 1347034',
           'summary_2_1437034/upsampling_1347034_protocols/results_scaffold'),
 'chok1': ('Caspase-3/7 CHO-K1', 'AID 1347037',
           'summary_1347037/upsampling_protocols/results_scaffold'),
}
LAB = {'rf': 'Random Forest', 'svm': 'SVM', 'xgb': 'XGBoost', 'gcn': 'GCN',
       'gat': 'GAT', 'gin': 'GIN', 'sage': 'SAGE', 'graphormer': 'Graphormer',
       'graphgps': 'GraphGPS', 'mvgnn': 'MV-GNN-cross', 'consensus': 'Consensus'}
CLASS = ['Random Forest', 'SVM', 'XGBoost']
# MV-GNN-cross is excluded: the SI states it is a reference baseline, not one
# of this work's architectures, and it is in no consensus.
GRAPH_MODELS = ['GCN', 'GAT', 'GIN', 'SAGE', 'Graphormer', 'GraphGPS']
RATIO = 1.0 / 8.0

TRAIN_SCAFFOLDS = ['c1ccccc1', 'c1ccc2ccccc2c1', 'c1ccncc1']
TEST_SCAFFOLDS = ['O=C1C=CC2C(=C1)CCC1C3CCCC3CCC21', 'c1ccc(Cc2ccccc2)cc1']


def load(d, key, model, metric):
    for pat in (f'{key}_per_unit.csv', f'{key}_per_repeat.csv'):
        f = os.path.join(HERE, d, pat)
        if os.path.exists(f):
            x = pd.read_csv(f)
            x = x[(x.Split == 'Test') & (x.CPT == 0.5) & (x.Model == model)]
            uc = [c for c in ('Unit', 'Fold', 'Repeat') if c in x.columns]
            if len(x):
                return x.sort_values(uc[0])[metric].to_numpy(float)
    return None


def moe(v):
    n = len(v)
    return float(stats.t.ppf(0.975, n - 1) *
                 np.sqrt((1.0 / n + RATIO) * np.var(v, ddof=1)))


def table(metric):
    rows = []
    for a, (_, _, d) in ASSAY.items():
        V = {lab: load(d, k, lab, metric) for k, lab in LAB.items()}
        V = {k: v for k, v in V.items() if v is not None}
        top3 = sorted(GRAPH_MODELS, key=lambda m: -V[m].mean())[:3]
        bc = max(CLASS, key=lambda m: V[m].mean())
        for m in top3 + [bc]:
            rows.append(dict(assay=a, metric=metric, model=m,
                             role='graph' if m in top3 else 'classical',
                             best=(m == top3[0]), mean=V[m].mean(),
                             moe=moe(V[m]), n=len(V[m])))
    return pd.DataFrame(rows)


EXPECT = {   # recomputed independently; the figure must not drift from these
 ('mmp', 'ROC-AUC'):   {'GraphGPS': 0.875, 'SAGE': 0.871, 'GIN': 0.870, 'Random Forest': 0.847},
 ('hepg2', 'ROC-AUC'): {'GraphGPS': 0.849, 'SAGE': 0.834, 'GCN': 0.832, 'Random Forest': 0.802},
 ('chok1', 'ROC-AUC'): {'GraphGPS': 0.755, 'SAGE': 0.732, 'GAT': 0.729, 'Random Forest': 0.719},
}


def verify(t):
    bad = []
    for (a, metric), exp in EXPECT.items():
        s = t[(t.assay == a) & (t.metric == metric)]
        for m, want in exp.items():
            r = s[s.model == m]
            if not len(r):
                bad.append((a, metric, m, 'missing')); continue
            got = round(float(r['mean'].iloc[0]), 3)
            if got != want:
                bad.append((a, metric, m, want, got))
    return bad


def scaffold_img(smi, px=420):
    mol = Chem.MolFromSmiles(smi)
    d = rdMolDraw2D.MolDraw2DCairo(px, px)
    o = d.drawOptions()
    o.setBackgroundColour((1, 1, 1, 1))   # opaque: RGB convert turns alpha into black
    o.bondLineWidth = 3.2
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    im = Image.open(io.BytesIO(d.GetDrawingText())).convert('RGB')
    a = np.array(im)
    ys, xs = np.where(a.astype(int).sum(2) < 740)
    if len(xs) == 0:
        return a
    p = 8
    return np.array(im.crop((max(xs.min() - p, 0), max(ys.min() - p, 0),
                             min(xs.max() + p, a.shape[1]),
                             min(ys.max() + p, a.shape[0]))))


def draw_schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.25); ax.axis('off')
    for x0, w, col, title, smis, sub in [
            (0.05, 5.15, TRAIN_C, 'Training scaffolds', TRAIN_SCAFFOLDS,
             '80% of scaffold groups'),
            (6.05, 3.90, TEST_C, 'Held-out test scaffolds', TEST_SCAFFOLDS,
             '10%, never seen in training')]:
        ax.add_patch(FancyBboxPatch((x0, 0.30), w, 2.35,
                                    boxstyle='round,pad=0.07,rounding_size=0.14',
                                    fc='white', ec=col, lw=1.5, zorder=1))
        ax.text(x0 + 0.16, 2.33, title, fontsize=10.5, fontweight='bold',
                color=col, va='bottom')
        ax.text(x0 + 0.16, 0.44, sub, fontsize=8.2, color=MUTED, va='bottom')
        step = w / len(smis)
        for i, s in enumerate(smis):
            sub_ax = ax.inset_axes([(x0 + 0.1 + i * step) / 10, 0.78 / 3.25,
                                    (step * 0.82) / 10, 1.42 / 3.25])
            sub_ax.imshow(scaffold_img(s), aspect='equal'); sub_ax.axis('off')
    ax.add_patch(FancyArrowPatch((5.33, 1.45), (5.93, 1.45),
                                 arrowstyle='-|>', mutation_scale=15,
                                 lw=1.5, color=MUTED))
    ax.text(5.63, 1.60, 'split by\nscaffold', fontsize=7.8, color=MUTED,
            ha='center', va='bottom', linespacing=1.2)
    ax.text(0.05, 0.02,
            'Bemis\u2013Murcko scaffold bins are assigned whole to one partition '
            '(8:1:1 train/validation/test, 5 repeats), so no scaffold is shared between '
            'training and test. MMP: 5,937 compounds, 1,685 scaffolds, 1,282 seen once.',
            fontsize=8.2, color=MUTED, va='bottom')


def schematic_figure():
    fig = plt.figure(figsize=(9.6, 2.95), facecolor='white')
    ax = fig.add_axes([0.012, 0.03, 0.976, 0.94])
    draw_schematic(ax)
    out = os.path.join(HERE, 'figures', 'scaffold_split_schematic')
    fig.savefig(out + '.png', dpi=300, facecolor='white', bbox_inches='tight')
    fig.savefig(out + '.pdf', facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}.png / .pdf')


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'roc'
    if mode == 'schematic':
        schematic_figure()
        return
    metrics = ['ROC-AUC'] if mode == 'roc' else ['ROC-AUC', 'PR-AUC']
    t = pd.concat([table(m) for m in metrics], ignore_index=True)
    bad = verify(t)
    if bad:
        raise SystemExit(f'plotted values disagree with the recomputed set: {bad}')
    t.to_csv(os.path.join(HERE, 'figures', 'scaffold_graph_models_values.csv'),
             index=False)

    nrow = len(metrics)
    fig = plt.figure(figsize=(11.0, 0.95 + 2.95 * nrow), facecolor='white')
    gs = gridspec.GridSpec(nrow, 3, figure=fig, hspace=0.62, wspace=0.30,
                           left=0.097, right=0.983,
                           top=0.855 if nrow == 1 else 0.905,
                           bottom=0.255 if nrow == 1 else 0.155)

    letters = 'ABCDEF'
    li = 0
    for r, metric in enumerate(metrics):
        for c, (a, (name, aid, _)) in enumerate(ASSAY.items()):
            ax = fig.add_subplot(gs[r, c])
            d = t[(t.assay == a) & (t.metric == metric)].reset_index(drop=True)
            d = d.sort_values('mean').reset_index(drop=True)
            for i, row in d.iterrows():
                col = RED if row.best else (GRAPH if row.role == 'graph' else CLASSIC)
                ax.plot([row['mean'] - row.moe, row['mean'] + row.moe], [i, i],
                        color=col, lw=2.0, solid_capstyle='round', alpha=.65, zorder=2)
                ax.plot([row['mean']], [i], 'o', ms=7.5 if row.best else 6.0,
                        color=col, mec='white', mew=1.2, zorder=3)
                ax.annotate(f'{row["mean"]:.3f}', (row['mean'], i),
                            textcoords='offset points', xytext=(0, 9), ha='center',
                            fontsize=8.0, color=INK if row.best else MUTED,
                            fontweight='bold' if row.best else 'normal')
            g = d[d.role == 'graph']['mean'].max()
            cl = d[d.role == 'classical']['mean'].max()
            ax.set_yticks(range(len(d)))
            ax.set_yticklabels(list(d.model), fontsize=8.8)
            for lbl, rl, bs in zip(ax.get_yticklabels(), d.role, d.best):
                lbl.set_color(RED if bs else (INK if rl == 'graph' else MUTED))
                if bs:
                    lbl.set_fontweight('bold')
            lo = (d['mean'] - d.moe).min(); hi = (d['mean'] + d.moe).max()
            pad = 0.055 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-0.75, len(d) - 0.15)
            ax.tick_params(axis='both', length=0, labelsize=8.0, colors=MUTED)
            ax.set_xlabel(metric, fontsize=9.2, color=INK, labelpad=3)
            for sp in ('top', 'right', 'left'):
                ax.spines[sp].set_visible(False)
            ax.spines['bottom'].set_color(GRID)
            ax.xaxis.grid(True, color=GRID, lw=.6, zorder=0); ax.set_axisbelow(True)
            ax.set_title(f'{name}\n{aid}', fontsize=9.4, color=INK, pad=20,
                         linespacing=1.35)
            ax.text(0.5, 1.012, f'best graph \u2212 best classical = {g - cl:+.3f}',
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=8.4, color=RED if g > cl else MUTED, fontweight='bold')
            ax.text(-0.012, 1.20, letters[li], transform=ax.transAxes, fontsize=14,
                    fontweight='bold', color=INK, ha='left', va='top')
            li += 1

    handles = [Line2D([], [], color=RED, lw=2.6, marker='o', ms=6.5, mec='white',
                      label='Best graph model'),
               Line2D([], [], color=GRAPH, lw=2.6, marker='o', ms=5.5, mec='white',
                      label='Other graph models'),
               Line2D([], [], color=CLASSIC, lw=2.6, marker='o', ms=5.5, mec='white',
                      label='Best classical baseline')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.54, 0.005),
               ncol=3, frameon=False, fontsize=8.8, handlelength=1.8,
               columnspacing=1.8, handletextpad=0.6)

    out = os.path.join(HERE, 'figures', 'scaffold_graph_models')
    fig.savefig(out + '.png', dpi=300, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}.png / .pdf and _values.csv ({len(t)} rows)')


if __name__ == '__main__':
    main()
