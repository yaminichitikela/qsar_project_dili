"""DILI held-out test set: Proxy-only against Proxy+DILI, both metrics.

Panel A: ROC-AUC. Panel B: PR-AUC. One row per model. Two forms:

  dumbbell  Proxy-only as a hollow marker, Proxy+DILI as a filled one, joined
            by a connector. The connector length is the fine-tuning effect and
            its direction shows whether the stage helped, which is the claim
            the text makes. Spread is reported in the companion table rather
            than drawn, so fifteen rows stay legible.

  interval  Both stages drawn as mean +/- SD bars, Proxy-only lighter and
            offset above Proxy+DILI. Honest about the spread, denser to read.

Colour encodes model family and is reinforced by the row label and the printed
value, so identity never rests on colour alone; stage is carried by marker fill
and position, never by a second set of hues. Both panels keep their own x
scale, because PR-AUC on a 70% positive test set sits far above ROC-AUC and the
two are not comparable by eye.

Usage: python make_dili_stage_figure.py [dumbbell|interval]
Outputs: figures/dili_stages_<form>.pdf/.png and figures/dili_stages_values.csv
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'figure_data'))
import dili_data as D

# five categorical hues (validated: lightness, chroma, CVD and normal-vision
# separation all pass) plus a neutral ink for the aggregate consensus rows
FAMILY_COLOR = {'Classical': '#2a78d6', 'GNN': '#1baf7a', 'GT': '#eb6834',
                'MV-GNN': '#4a3aa7', 'Baseline': '#a8871f', 'Consensus': '#3f3e3b'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#dcdcd8'
# emphasis for the leading models: a red distinct from the GT orange, always
# paired with bold weight and a box so identity never rests on colour alone
HI = '#c2252c'
# Mark exactly one leader per stage per panel -- the model that actually tops
# that stage -- so the emphasis never lands on a model that merely placed well.
TITLE = 'FDA-approved DILI held-out test set (Held-223)'
METRICS = [('ROC-AUC', 'A'), ('PR-AUC', 'B')]
PAD = 0.055
OFF = 0.19          # vertical offset between the two stages


def main():
    form = sys.argv[1] if len(sys.argv) > 1 else 'dumbbell'
    if form not in ('dumbbell', 'interval'):
        raise SystemExit("form must be 'dumbbell' or 'interval'")
    protos = dict(D.PROTOS)
    proto = sys.argv[2] if len(sys.argv) > 2 else D.PROTOS[0][0]
    if proto not in protos:
        raise SystemExit(f'unknown protocol {proto!r}; pick from {list(protos)}')
    ptitle = protos[proto]
    t = D.table(proto)
    bad = D.verify(t)
    if bad:
        raise SystemExit(f'per-unit check failed: {bad[:5]}')
    t.to_csv(os.path.join(HERE, 'figures',
                          f'dili_stages_{proto}_values.csv'), index=False)

    lead1, lead2 = {}, {}
    for me, _ in METRICS:
        d_ = t[t.metric == me]
        lead1[me] = d_[d_.stage == 1].set_index('model')['mean'].idxmax()
        lead2[me] = d_[d_.stage == 2].set_index('model')['mean'].idxmax()
    starred = set(lead1.values()) | set(lead2.values())

    models = [m for m in D.ORDER if m in set(t.model)]
    ypos = {m: len(models) - 1 - i for i, m in enumerate(models)}
    nrep = int(t.n.max())

    # wide and short: 15 rows still clear, but the figure claims far less
    # vertical space on the page than a square-ish panel would
    fig = plt.figure(figsize=(11.4, 5.0))
    gs = fig.add_gridspec(1, 2, wspace=0.07, left=0.158, right=0.988,
                          top=0.885, bottom=0.140)

    for j, (metric, letter) in enumerate(METRICS):
        ax = fig.add_subplot(gs[0, j])
        d = t[t.metric == metric]
        s1 = d[d.stage == 1].set_index('model')
        s2 = d[d.stage == 2].set_index('model')
        lo_v = min(s1.lo.min(), s2.lo.min()) if form == 'interval' else \
            min(s1['mean'].min(), s2['mean'].min())
        hi_v = max(s1.hi.max(), s2.hi.max()) if form == 'interval' else \
            max(s1['mean'].max(), s2['mean'].max())
        span = hi_v - lo_v
        best = lead2[metric]
        first_stage1 = lead1[metric]

        for m in models:
            y = ypos[m]
            c = FAMILY_COLOR[D.FAMILY[m]]
            a, b = s1.loc[m, 'mean'], s2.loc[m, 'mean']
            if form == 'dumbbell':
                ax.plot([a, b], [y, y], color=c, lw=2.0, alpha=.45,
                        solid_capstyle='round', zorder=2)
                ax.plot([a], [y], 'o', ms=5.2, mfc='white', mec=c, mew=1.5,
                        zorder=3)
                ax.plot([b], [y], 'o', ms=5.8 if m == best else 5.2, color=c,
                        mec='white', mew=1.1, zorder=4)
                lead = (m == best)
                ax.annotate(
                    f'{b:.3f}', (b, y), textcoords='offset points',
                    xytext=(0, 7.4), ha='center', fontsize=8.6 if lead else 8.0,
                    color=HI if lead else MUTED,
                    fontweight='bold' if lead else 'normal',
                    bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=HI,
                              lw=0.9, alpha=.95) if lead else None,
                    zorder=6)
                if m == first_stage1 and m != best:
                    # the Proxy-only leader carries its own boxed value, set to
                    # the left of its marker so it cannot collide with the
                    # Proxy+DILI label above or with the row beneath
                    ax.annotate(
                        f'{a:.3f}', (a, y), textcoords='offset points',
                        xytext=(-8, 0), ha='right', va='center', fontsize=8.6,
                        color=HI, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.22', fc='white',
                                  ec=HI, lw=0.9, ls=(0, (2, 1.4)), alpha=.95),
                        zorder=6)
            else:
                for st, src, yo, alpha, filled in ((1, s1, OFF, .40, False),
                                                   (2, s2, -OFF, .80, True)):
                    mu, lo, hi = src.loc[m, 'mean'], src.loc[m, 'lo'], src.loc[m, 'hi']
                    ax.plot([lo, hi], [y + yo, y + yo], color=c, lw=1.9,
                            alpha=alpha, solid_capstyle='round', zorder=2)
                    ax.plot([mu], [y + yo], 'o', ms=5.6,
                            color=c if filled else 'white', mec=c,
                            mew=1.1 if filled else 1.6, zorder=3)
                ax.annotate(f'{b:.3f}', (b, y - OFF), textcoords='offset points',
                            xytext=(0, 7), ha='center', fontsize=7.8,
                            color=INK if m == best else MUTED,
                            fontweight='bold' if m == best else 'normal')

        for s in D.SEPS:
            ax.axhline(len(models) - 1 - s - 0.5, color=GRID, lw=.7,
                       ls=(0, (1, 3)), zorder=1)
        ax.set_ylim(-0.85, len(models) - 0.15)
        ax.set_xlim(lo_v - PAD * span, hi_v + PAD * span)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(
            [m for m in models[::-1]] if j == 0 else [], fontsize=8.6, color=INK)
        ax.tick_params(axis='both', length=0, labelsize=7.6, colors=MUTED)
        if j == 0:
            for lbl, m in zip(ax.get_yticklabels(), models[::-1]):
                if m in starred:
                    lbl.set_color(HI); lbl.set_fontweight('bold')
        ax.set_xlabel(metric, fontsize=9.4, color=INK, labelpad=3)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.xaxis.grid(True, color=GRID, lw=.6, zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.0, 1.028, letter, transform=ax.transAxes, fontsize=11.5,
                fontweight='bold', color=INK, ha='left', va='bottom')

    # No in-figure title or subtitle: the assay, protocol, repeat count and
    # marker key all live in the caption, so the panel area carries only data.
    handles = [Line2D([], [], color=c, lw=2.4, label=f)
               for f, c in FAMILY_COLOR.items()]
    handles += [Line2D([], [], color=MUTED, marker='o', ls='none', mfc='white',
                       mew=1.6, ms=6, label='Proxy-only'),
                Line2D([], [], color=MUTED, marker='o', ls='none', ms=6,
                       label='Proxy+DILI')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.573, 1.006),
               ncol=8, frameon=False, fontsize=8.0, handlelength=1.4,
               columnspacing=1.5, handletextpad=0.5)

    out = os.path.join(HERE, 'figures', f'dili_stages_{proto}_{form}')
    fig.savefig(out + '.png', dpi=300, facecolor='white')
    fig.savefig(out + '.pdf', facecolor='white')
    plt.close(fig)
    print(f'Saved {out}.png / .pdf  ({len(models)} models, {nrep} repeats)')


if __name__ == '__main__':
    main()
