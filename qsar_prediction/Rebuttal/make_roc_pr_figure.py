"""Single-assay, single-protocol results: ROC-AUC and PR-AUC side by side.

Panel A: ROC-AUC. Panel B: PR-AUC. One row per model, point estimate with a
95% interval. No ranking panel.

WHY BOTH METRICS
----------------
Both summary assays are imbalanced (MMP: 857 active of 5,937 scored). ROC-AUC
is largely insensitive to that imbalance; PR-AUC is not, and is driven by the
positive class the screen is actually looking for. Showing them together says
whether a model's standing survives the metric that matters for a rare-hit
screen. PR-AUC is necessarily lower than ROC-AUC, so each panel carries its
own x scale and the two are not compared by eye across panels.

INTERVALS
---------
Nadeau-Bengio corrected resampled t interval,
mean +/- t(0.975, n-1) * sqrt((1/n + n_test/n_train) * var), identical to
compute_ci_significance.py, because the folds share training data and the naive
1/n variance would be anticonservative. The loader cross-checks this against
results/ci_summary.csv before plotting and refuses to draw if it disagrees.
Each axis spans the full interval range, so nothing is truncated.

Colours encode model family and are reinforced by row labels and printed
values, so identity never rests on colour alone.

Usage: python make_roc_pr_figure.py [mmp|hepg2] [protocol]
Outputs: figures/<assay>_roc_pr_<protocol>.pdf/.png and _values.csv
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'figure_data'))

FAMILY_COLOR = {'Classical': '#2a78d6', 'GT': '#eb6834', 'GNN': '#1baf7a',
                'MV-GNN': '#4a3aa7', 'Consensus': '#3f3e3b'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#dcdcd8'
ASSAY_TITLE = {'mmp': 'Mitochondrial membrane potential (AID 720637)',
               'hepg2': 'Caspase-3/7 HepG2 (AID 654+1347034)'}
METRICS = [('ROC-AUC', 'A'), ('PR-AUC', 'B')]
SEPS = [2, 6, 8, 9]          # dotted rule after these rows -> family breaks
PAD = 0.045                  # axis padding, fraction of the interval span


def main():
    assay = sys.argv[1] if len(sys.argv) > 1 else 'mmp'
    D = __import__(f'{assay}_data')
    ptitle = dict(D.PROTOS)
    proto = sys.argv[2] if len(sys.argv) > 2 else D.PROTOS[0][0]
    if proto not in ptitle:
        raise SystemExit(f'unknown protocol {proto!r}; pick from {list(ptitle)}')

    t = D.table()
    bad = D.verify(t)
    if bad:
        raise SystemExit(f'interval check against ci_summary.csv failed: {bad}')
    t[['lo', 'hi']] = t[['lo', 'hi']].clip(0.0, 1.0)
    t = t[t.proto == proto]
    t.to_csv(os.path.join(HERE, 'figures',
                          f'{assay}_roc_pr_{proto}_values.csv'), index=False)

    models = [m for m in D.ORDER if m in set(t.model)]
    ypos = {m: len(models) - 1 - i for i, m in enumerate(models)}
    nunit = int(t.n.max())

    fig = plt.figure(figsize=(10.4, 5.9))
    gs = fig.add_gridspec(1, 2, wspace=0.075, left=0.152, right=0.986,
                          top=0.805, bottom=0.115)

    for j, (metric, letter) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j])
        d = t[t.metric == metric].set_index('model')
        span = d.hi.max() - d.lo.min()
        lo_ax = max(0.0, d.lo.min() - PAD * span)
        hi_ax = min(1.0, d.hi.max() + PAD * span)
        best = d['mean'].idxmax()
        for m in models:
            y = ypos[m]
            mu, lo, hi = d.loc[m, 'mean'], d.loc[m, 'lo'], d.loc[m, 'hi']
            c = FAMILY_COLOR[D.FAMILY[m]]
            ax.plot([lo, hi], [y, y], color=c, lw=1.9,
                    solid_capstyle='round', alpha=.72, zorder=2)
            ax.plot([mu], [y], 'o', ms=6.4 if m == best else 5.8, color=c,
                    mec='white', mew=1.1, zorder=3)
            ax.annotate(f'{mu:.3f}', (mu, y), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=7.4,
                        color=INK if m == best else MUTED,
                        fontweight='bold' if m == best else 'normal')
        for s in SEPS:
            ax.axhline(len(models) - 1 - s - 0.5, color=GRID, lw=.7,
                       ls=(0, (1, 3)), zorder=1)
        ax.set_ylim(-0.8, len(models) - 0.2)
        ax.set_xlim(lo_ax, hi_ax)
        ax.set_yticks(range(len(models)))
        if j == 0:
            ax.set_yticklabels(
                [m + ('†' if m in D.CORRECTED else '') for m in models[::-1]],
                fontsize=9, color=INK)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis='both', length=0, labelsize=8.4, colors=MUTED)
        ax.set_xlabel(metric, fontsize=9.8, color=INK, labelpad=4)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.xaxis.grid(True, color=GRID, lw=.6, zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.0, 1.028, letter, transform=ax.transAxes, fontsize=13,
                fontweight='bold', color=INK, ha='left', va='bottom')

    fig.text(0.152, 0.955, ASSAY_TITLE.get(assay, assay), fontsize=11.5,
             color=INK, ha='left', va='bottom')
    fig.text(0.152, 0.902,
             f'{ptitle[proto]}  ·  {nunit} folds  ·  '
             '95% corrected resampled t interval',
             fontsize=9, color=MUTED, ha='left', va='bottom')
    handles = [Line2D([], [], color=c, lw=2.4, label=f)
               for f, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.986, 0.995),
               ncol=5, frameon=False, fontsize=9, handlelength=1.7,
               columnspacing=1.6, handletextpad=0.6)

    out = os.path.join(HERE, 'figures', f'{assay}_roc_pr_{proto}')
    fig.savefig(out + '.png', dpi=300, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}.png / .pdf and _values.csv  ({len(t)} rows)')


if __name__ == '__main__':
    main()
