"""ACS table-of-contents graphic: DILI prediction, FDA-approved drugs.

THE CLAIM
---------
Graph representations shift the operating point toward catching hepatotoxic
drugs (sensitivity 70% vs 42%) at the cost of specificity, and the consensus
is the only configuration that lifts balanced accuracy above either family.

WHY NOT F1
----------
The Held-223 test set is 156 hepatotoxic of 223, i.e. 70.0% positive. An
all-positive classifier scores F1 = 0.823, which is higher than every model in
the study (best: GIN, 0.733). F1 on this test set therefore cannot support a
claim about any model, and the large graph-vs-classical F1 gap (0.719 vs
0.536) is an operating-point artefact: balanced accuracy differs by 0.1 points
and ROC-AUC by 0.003 between those two families. Sensitivity, specificity and
balanced accuracy are plotted instead, because together they show the trade
honestly and balanced accuracy has a meaningful 50% chance line.

DATA
----
results/dili_corrected_gt_values.csv, Stage 2 (Proxy+DILI), cutoff 0.50,
corrected graph-transformer architectures, 5 repeated random splits of the
clinical pool against the fixed Held-223 external test set. Values reproduce
tables_dili_repeated_split.tex (tab:dili_rr_s2_cpt50).

Bars are means over the models in each family. Error bars are the spread
across those models (SD), except Consensus, a single model, where the bar is
its SD over the 5 repeats. Both are stated in the caption note, not conflated.

Canvas is ACS spec: 3.25 x 1.75 in.

Usage: python make_dili_toc_graphic.py
Outputs: figures/dili_toc_graphic.png (600 dpi) and .pdf, plus _values.csv
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'results', 'dili_corrected_gt_values.csv')

CLASS = ['RF', 'SVM', 'XGB']
GRAPH = ['GCN', 'GAT', 'GIN', 'SAGE', 'Graphormer', 'GraphGPS']
FAMILIES = [('Classical', CLASS), ('Graph', GRAPH), ('Consensus', ['Consensus'])]
# (column, label, colour, trivial baseline on Held-223 or None)
# Baselines are no longer DRAWN (values are printed on the bars instead), but
# they are still asserted in verify() and belong in the figure caption:
#   F1        0.823 = an all-positive classifier (156 of 223 are positive)
#   PR-AUC    0.700 = a random ranker, i.e. the positive prevalence
#   ROC-AUC   0.500 = a random ranker
#   Bal. acc. 0.500 = any constant classifier
# Sensitivity and specificity have no single meaningful baseline (each is
# trivially maximised by a constant prediction), so they carry none and are
# read as a pair.
METRICS = [('sens', 'Sens.', '#0f2e4d', None),
           ('spec', 'Spec.', '#2a6f9e', None),
           ('ccr', 'Bal. acc.', '#6aa9c9', 0.500),
           ('f1', 'F1', '#4a5560', 0.8232),
           ('roc_auc', 'ROC-AUC', '#78838f', 0.500),
           ('pr_auc', 'PR-AUC', '#a8b2bd', 0.6996)]
HIGHLIGHT = 'Graph'          # family boxed in red
BOX = '#c62828'


def _label_colour(hex_colour):
    """White on dark bars, ink on light ones, by relative luminance."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
           for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return '#ffffff' if lum < 0.42 else '#0b0b0b'
PCT = {'sens', 'spec', 'ccr'}          # stored as percentages, plotted on 0-1
# the CSV abbreviates the SD columns for the two AUCs, so map them explicitly
SD_COL = {'sens': 'sens_sd', 'spec': 'spec_sd', 'ccr': 'ccr_sd',
          'f1': 'f1_sd', 'roc_auc': 'roc_sd', 'pr_auc': 'pr_sd'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#dcdcd8'

N_POS, N_TOT = 156, 223          # Held-223 class counts, verified from the file
EXPECT = {'Classical': (0.417, 0.719, 0.568, 0.536, 0.610, 0.770),
          'Graph':     (0.700, 0.434, 0.567, 0.719, 0.613, 0.775),
          'Consensus': (0.638, 0.555, 0.596, 0.697, 0.636, 0.790)}


def table():
    d = pd.read_csv(CSV)
    s = d[(d.stage == 'Stage 2 (Proxy+DILI)') & (d.cpt == 0.5)].set_index('model')
    rows = []
    for fam, models in FAMILIES:
        r = s.loc[models]
        for key, lab, _, base in METRICS:
            k = 100.0 if key in PCT else 1.0
            mean = float(r[key].mean()) / k
            err = (float(s.loc[models[0], SD_COL[key]]) if len(models) == 1
                   else float(r[key].std(ddof=1))) / k
            rows.append(dict(family=fam, metric=lab, mean=mean, err=err,
                             baseline=base, n_models=len(models)))
    return pd.DataFrame(rows)


def verify(t):
    bad = []
    for fam, want in EXPECT.items():
        got = tuple(round(float(t[(t.family == fam) & (t.metric == m)]['mean'].iloc[0]), 3)
                    for _, m, _, _ in METRICS)
        if got != want:
            bad.append((fam, want, got))
    g = t[(t.family == 'Graph') & (t.metric == 'Sens.')]['mean'].iloc[0]
    c = t[(t.family == 'Classical') & (t.metric == 'Sens.')]['mean'].iloc[0]
    if not g > c + 0.20:
        bad.append(('sensitivity gap collapsed', c, g))
    # the whole point of the baselines: F1 must still sit under 0.8232
    if t[t.metric == 'F1']['mean'].max() >= 0.8232:
        bad.append(('a model now beats the all-positive F1 baseline', None))
    return bad


def main():
    t = table()
    bad = verify(t)
    if bad:
        raise SystemExit(f'DILI TOC numbers changed: {bad}')
    t.to_csv(os.path.join(HERE, 'figures', 'dili_toc_graphic_values.csv'), index=False)

    fig = plt.figure(figsize=(3.25, 1.75), facecolor='white')
    ax = fig.add_axes([0.105, 0.155, 0.88, 0.545])

    xs = np.arange(len(FAMILIES))
    w = 0.142
    for i, (key, lab, col, base) in enumerate(METRICS):
        vals = [t[(t.family == f) & (t.metric == lab)]['mean'].iloc[0] for f, _ in FAMILIES]
        errs = [t[(t.family == f) & (t.metric == lab)]['err'].iloc[0] for f, _ in FAMILIES]
        pos = xs + (i - (len(METRICS) - 1) / 2) * w
        ax.bar(pos, vals, w * 0.88, color=col, label=lab, zorder=3,
               edgecolor='white', linewidth=0.3)
        ax.errorbar(pos, vals, yerr=errs, fmt='none', ecolor=INK, alpha=.5,
                    elinewidth=0.6, capsize=0.9, capthick=0.6, zorder=5)
        for x, v in zip(pos, vals):
            # rotated inside the bar: 18 bars are too narrow for horizontal
            # labels, and this keeps every value clear of the error bars
            ax.text(x, v * 0.46, f'{v:.2f}'.lstrip('0'), ha='center', va='center',
                    rotation=90, fontsize=4.9, color=_label_colour(col),
                    fontweight='bold', zorder=9)

    hi = [f for f, _ in FAMILIES].index(HIGHLIGHT)
    half = len(METRICS) / 2 * w + 0.052
    top = max(t[t.family == HIGHLIGHT]['mean'] + t[t.family == HIGHLIGHT]['err']) + 0.055
    ax.add_patch(Rectangle((hi - half, 0), 2 * half, top, fill=False,
                           edgecolor=BOX, lw=1.0, zorder=7, clip_on=False))

    ax.set_xticks(xs)
    ax.set_xticklabels([f for f, _ in FAMILIES], fontsize=6.6, color=INK)

    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(['0', '.25', '.50', '.75', '1'])
    ax.tick_params(axis='both', length=0, labelsize=5.8, colors=MUTED, pad=1.2)
    # after tick_params, which would otherwise reset the colour
    for lbl, (f, _) in zip(ax.get_xticklabels(), FAMILIES):
        if f == HIGHLIGHT:
            lbl.set_color(BOX); lbl.set_fontweight('bold')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('bottom', 'left'):
        ax.spines[sp].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, lw=.45, zorder=0); ax.set_axisbelow(True)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=6,
              frameon=False, fontsize=5.1, handlelength=0.7,
              columnspacing=0.6, handletextpad=0.28)

    fig.text(0.105, 0.905, 'Graph models catch more hepatotoxic drugs',
             fontsize=7.4, color=INK, ha='left', va='bottom', fontweight='bold')
    # a COUNT, not a percentage: the prevalence is 70.0% and the Graph
    # sensitivity bar is 0.70, so a percentage here reads as a score.
    fig.text(0.105, 0.848,
             f'FDA-approved drugs \u00b7 external test set \u00b7 '
             f'{N_POS} of {N_TOT} hepatotoxic',
             fontsize=5.6, color=MUTED, ha='left', va='bottom')

    out = os.path.join(HERE, 'figures', 'dili_toc_graphic')
    fig.savefig(out + '.png', dpi=600, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}.png / .pdf  (3.25 x 1.75 in, ACS TOC spec)')


if __name__ == '__main__':
    main()
