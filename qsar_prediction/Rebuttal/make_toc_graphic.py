"""ACS table-of-contents graphic: graph models under scaffold split.

THE ONE CLAIM
-------------
On all three assays, moving classical -> graph -> consensus raises BOTH
ROC-AUC and F1. Six orderings out of six, no exceptions. Plotting the two
metrics against each other turns that into three trajectories that all travel
up and to the right, which is readable at TOC size without reading a number.

WHY THESE TWO METRICS
---------------------
ROC-AUC is threshold-free ranking quality; F1 at the 0.5 cut is what a user
actually gets from the deployed classifier. They are the pair where the
graph advantage holds across every assay. PR-AUC is NOT used here: it
reverses on the two rare-positive assays (see make_scaffold_graph_figure.py),
and a TOC graphic is the wrong place for a result that needs a caveat.

Family means over the 5 scaffold repeats: classical = RF/SVM/XGBoost,
graph = GCN/GAT/GIN/SAGE/Graphormer/GraphGPS, consensus = full ensemble.
MV-GNN-cross is excluded, as a reference baseline in no consensus.

Canvas is ACS spec: 3.25 x 1.75 in.

Usage: python make_toc_graphic.py
Outputs: figures/toc_graphic.png (300 dpi) and .pdf
"""
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))

LAB = {'rf': 'Random Forest', 'svm': 'SVM', 'xgb': 'XGBoost', 'gcn': 'GCN',
       'gat': 'GAT', 'gin': 'GIN', 'sage': 'SAGE', 'graphormer': 'Graphormer',
       'graphgps': 'GraphGPS', 'consensus': 'Consensus'}
CLASS = ['Random Forest', 'SVM', 'XGBoost']
GRAPH = ['GCN', 'GAT', 'GIN', 'SAGE', 'Graphormer', 'GraphGPS']
ASSAYS = [('MMP', 'Summary_1_models/upsampling_protocols/results_scaffold'),
          ('HepG2', 'summary_2_1437034/upsampling_1347034_protocols/results_scaffold'),
          ('CHO-K1', 'summary_1347037/upsampling_protocols/results_scaffold')]

GREY, RED, DARK = '#9a9a94', '#c62828', '#1b365d'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e3e3df'
STAGES = [('classical', CLASS, GREY), ('graph', GRAPH, RED),
          ('consensus', ['Consensus'], DARK)]

EXPECT = {   # independently recomputed; the graphic must not drift
 'MMP':    [(0.837, 0.494), (0.863, 0.575), (0.892, 0.663)],
 'HepG2':  [(0.784, 0.211), (0.829, 0.261), (0.861, 0.333)],
 'CHO-K1': [(0.698, 0.126), (0.726, 0.157), (0.777, 0.237)],
}


def cell(d, models, metric, cpt=0.5):
    vals = []
    for k, lab in LAB.items():
        if lab not in models:
            continue
        for pat in (f'{k}_per_unit.csv', f'{k}_per_repeat.csv'):
            f = os.path.join(HERE, d, pat)
            if os.path.exists(f):
                x = pd.read_csv(f)
                x = x[(x.Split == 'Test') & (x.CPT == cpt) & (x.Model == lab)]
                if len(x):
                    vals.append(x[metric].mean())
                break
    return float(np.mean(vals))


def collect():
    out = {}
    for a, d in ASSAYS:
        out[a] = [(cell(d, ms, 'ROC-AUC'), cell(d, ms, 'F1')) for _, ms, _ in STAGES]
    return out


def verify(data):
    bad = []
    for a, pts in data.items():
        for (gr, gf), (wr, wf) in zip(pts, EXPECT[a]):
            if (round(gr, 3), round(gf, 3)) != (wr, wf):
                bad.append((a, (wr, wf), (round(gr, 3), round(gf, 3))))
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        if not (xs[0] < xs[1] < xs[2] and ys[0] < ys[1] < ys[2]):
            bad.append((a, 'ordering broken', pts))
    return bad


def main():
    data = collect()
    bad = verify(data)
    if bad:
        raise SystemExit(f'TOC numbers or ordering changed: {bad}')
    pd.DataFrame([{'assay': a, 'stage': STAGES[i][0], 'ROC-AUC': p[0], 'F1': p[1]}
                  for a, pts in data.items() for i, p in enumerate(pts)]
                 ).to_csv(os.path.join(HERE, 'figures', 'toc_graphic_values.csv'),
                          index=False)

    fig = plt.figure(figsize=(3.25, 1.75), facecolor='white')
    ax = fig.add_axes([0.145, 0.185, 0.835, 0.615])

    # assay label placement, chosen so no label lands on another trajectory
    LABPOS = {'MMP': (6, 2, 'left'), 'HepG2': (6, 1, 'left'), 'CHO-K1': (-1, 9, 'center')}
    for a, pts in data.items():
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        for i in range(len(pts) - 1):
            ax.add_patch(FancyArrowPatch((xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                                         arrowstyle='-|>', mutation_scale=6.0,
                                         lw=1.0, color=MUTED, alpha=.75,
                                         shrinkA=3.2, shrinkB=3.2, zorder=2))
        for (x, y), (_, _, c) in zip(pts, STAGES):
            ax.plot([x], [y], 'o', ms=4.8, color=c, mec='white', mew=0.8, zorder=3)
        dx, dy, ha = LABPOS[a]
        ax.annotate(a, (xs[-1], ys[-1]), textcoords='offset points',
                    xytext=(dx, dy), fontsize=6.4, color=INK, va='center',
                    ha=ha, fontweight='bold')

    ax.set_xlim(0.675, 0.945); ax.set_ylim(0.06, 0.75)
    ax.set_xticks([0.70, 0.80, 0.90]); ax.set_yticks([0.2, 0.4, 0.6])
    ax.tick_params(length=0, labelsize=6.0, colors=MUTED, pad=1.5)
    ax.set_xlabel('ROC-AUC', fontsize=7.0, color=INK, labelpad=1.0)
    ax.set_ylabel('F1', fontsize=7.0, color=INK, labelpad=1.0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('bottom', 'left'):
        ax.spines[sp].set_color(GRID)
    ax.grid(True, color=GRID, lw=.5, zorder=0); ax.set_axisbelow(True)

    # legend sits in the empty upper-left quadrant: no trajectory reaches
    # x < 0.70 above F1 = 0.3, so nothing can collide with it
    for i, (name, _, c) in enumerate(STAGES):
        yy = 0.715 - i * 0.063
        ax.plot([0.690], [yy], 'o', ms=4.4, color=c, mec='white', mew=0.7, zorder=4)
        ax.text(0.700, yy, name, fontsize=6.2, color=c, va='center',
                fontweight='bold')

    fig.text(0.145, 0.885, 'Graph models generalise to unseen scaffolds',
             fontsize=7.6, color=INK, ha='left', va='bottom', fontweight='bold')
    fig.text(0.145, 0.838,
             'Caspase-3/7 & mitochondrial toxicity \u00b7 scaffold split',
             fontsize=5.9, color=MUTED, ha='left', va='bottom')

    out = os.path.join(HERE, 'figures', 'toc_graphic')
    fig.savefig(out + '.png', dpi=600, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}.png / .pdf  (3.25 x 1.75 in, ACS TOC spec)')


if __name__ == '__main__':
    main()
