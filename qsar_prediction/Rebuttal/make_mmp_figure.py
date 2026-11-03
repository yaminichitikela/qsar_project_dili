"""Builds the MMP main-results figure: a forest plot of ROC-AUC with intervals
across the three evaluation protocols, plus a critical-difference diagram.

WHY THESE TWO PANELS
--------------------
Panel A (forest plot) shows level AND uncertainty side by side, so a reader can
see immediately that the best individual models sit inside one another's
intervals. That is the honest picture, and it is what the AUC-with-intervals
literature recommends over a bare bar chart of means.

Panel B (critical-difference diagram, Demsar 2006) answers the question a bar
chart cannot: across all 20 split units, which models rank higher, and which
differences are too small to call? Models joined by a horizontal bar are not
significantly different under a Nemenyi post-hoc test. This is the standard
form for "many models, several evaluation settings", and it keeps the figure
from overclaiming -- which matters here because the rebuttal (General Response
G5, Editor comment E1) already withdrew claims of individual-model superiority.

Together they support the defensible statement: graph-based models occupy the
top ranks, the consensus ensemble is ahead of every individual model, and the
individual models are not separable from one another.

RANKING UNIT
------------
The Nemenyi test treats each split unit (5 repeated random splits + 10 CV folds
+ 5 scaffold repeats = 20) as an observation. Units within one protocol share
data, so they are not fully independent datasets in Demsar's original sense;
the caption says so. The alternative -- three observations, one per protocol --
has no power at all.

Colours encode model family and are reinforced by row labels and printed values,
so identity never rests on colour alone (the aqua slot sits below 3:1 contrast
on a light surface, which the data-viz relief rule permits only with visible
labels).

Usage: python make_mmp_figure.py
Outputs: figures/mmp_main_results.pdf and .png
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/private/tmp/claude-501/-Users-yamini-Desktop-projects-qsar-casepase/'
                   'a3397216-f8e3-4ef4-b1d6-bf13e2b7665b/scratchpad')
from gen5 import ASSAY, PROTO, collect  # shared loaders, already verified vs published

# ── validated categorical palette (slots 1/2/3/7 of the reference palette;
#    passes all-pairs CVD and normal-vision floors in light mode) ────────────
FAMILY_COLOR = {
    'Classical': '#2a78d6',   # blue
    'GT': '#eb6834',          # orange
    'GNN': '#1baf7a',         # aqua
    'MV-GNN': '#4a3aa7',      # violet
    'Consensus': '#3f3e3b',   # neutral ink -- a summary row, not a peer family
}
FAMILY = {'Random Forest': 'Classical', 'SVM': 'Classical', 'XGBoost': 'Classical',
          'GCN': 'GNN', 'GAT': 'GNN', 'GIN': 'GNN', 'SAGE': 'GNN',
          'Graphormer': 'GT', 'GraphGPS': 'GT',
          'MV-GNN$^{\\text{cross}}$': 'MV-GNN', 'Consensus (Full)': 'Consensus'}
SHORT = {'Random Forest': 'Random Forest', 'MV-GNN$^{\\text{cross}}$': 'MV-GNN-cross',
         'Consensus (Full)': 'Consensus'}
ORDER = ['Random Forest', 'SVM', 'XGBoost', 'GCN', 'GAT', 'GIN', 'SAGE',
         'Graphormer', 'GraphGPS', 'MV-GNN$^{\\text{cross}}$', 'Consensus (Full)']
PROTOS = [('random_repeat', 'Repeated random split\n(5 repeats)'),
          ('loo', '10-fold cross-validation'),
          ('scaffold', 'Scaffold split\n(5 repeats)')]
CORRECTED = {'Graphormer', 'GraphGPS', 'Consensus (Full)'}

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#dcdcd8'
# Nemenyi critical values q_0.05 (Demsar 2006, Table 5), indexed by model count
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
       9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def per_unit_auc():
    """{protocol: DataFrame(unit x model) of test ROC-AUC at CPT 0.5}."""
    REB = HERE
    out = {}
    for proto, _ in PROTOS:
        A, P = ASSAY['mmp'], PROTO[proto]
        base = f'{REB}/{A[P["where"]]}' + (f'/{P["sub"]}' if P['sub'] else '')
        cor = f'{REB}/graph_transformer_corrected/results/mmp/{proto}'
        cols = {}
        for key, lab in [('rf', 'Random Forest'), ('svm', 'SVM'), ('xgb', 'XGBoost'),
                         ('gcn', 'GCN'), ('gat', 'GAT'), ('gin', 'GIN'), ('sage', 'SAGE'),
                         ('mvgnn', 'MV-GNN$^{\\text{cross}}$')]:
            f = f'{base}/{key}_{P["suf"]}_{P["pu"]}.csv'
            if os.path.exists(f):
                d = pd.read_csv(f)
                d = d[(d.Split == 'Test') & (d.CPT == 0.5)]
                cols[lab] = d['ROC-AUC'].values
        # corrected transformers and consensus
        pu = f'{cor}/per_unit_all.csv'
        if os.path.exists(pu):
            d = pd.read_csv(pu)
            for lab in ('Graphormer', 'GraphGPS'):
                g = d[(d.Model == lab) & (d.CPT == 0.5)].sort_values('Unit')
                if len(g):
                    cols[lab] = g['ROC-AUC'].values
        cm = f'{cor}/consensus_metrics.csv'
        if os.path.exists(cm):
            d = pd.read_csv(cm)
            g = d[(d.Model == 'Consensus') & (d.CPT == 0.5)].sort_values('Unit')
            if len(g):
                cols['Consensus (Full)'] = g['ROC-AUC'].values
        n = min(len(v) for v in cols.values())
        out[proto] = pd.DataFrame({k: v[:n] for k, v in cols.items()})
    return out


def summaries():
    """{protocol: DataFrame with Model, AUC, AUCm} from the verified loader."""
    return {p: collect('mmp', p)[0] for p, _ in PROTOS}


def main():
    units = per_unit_auc()
    summ = summaries()
    # every model that appears anywhere -> rows of panel A
    models = [m for m in ORDER if any(m in units[p].columns for p, _ in PROTOS)]
    # ranks need one consistent model set across protocols. MV-GNN-cross was not
    # run on the repeated random splits (those covered nine models), so it cannot
    # be ranked against the others without dropping a whole protocol; it is shown
    # in panel A and excluded from panel B, which the caption states.
    rank_models = [m for m in ORDER if all(m in units[p].columns for p, _ in PROTOS)]
    excluded = [m for m in models if m not in rank_models]

    ranks = []
    for proto, _ in PROTOS:
        ranks.append(units[proto][rank_models].rank(axis=1, ascending=False))
    R = pd.concat(ranks, ignore_index=True)
    N, k = len(R), len(rank_models)
    mean_rank = R.mean().sort_values()
    chi2, p_fried = stats.friedmanchisquare(*[R[m].values for m in rank_models])
    cd = Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))

    lows, highs = [], []
    for proto, _ in PROTOS:
        t = summ[proto]
        lows.append((t.AUC - t.AUCm).min())
        highs.append((t.AUC + t.AUCm).max())
    xlo, xhi = min(lows) - 0.015, max(highs) + 0.015

    fig = plt.figure(figsize=(7.4, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.05, 1], hspace=0.30, wspace=0.10,
                          left=0.205, right=0.985, top=0.925, bottom=0.075)

    # ── Panel A: forest plot, one column per protocol ───────────────────────
    ypos = {m: len(models) - 1 - i for i, m in enumerate(models)}
    for j, (proto, title) in enumerate(PROTOS):
        ax = fig.add_subplot(gs[0, j])
        t = summ[proto].set_index('Model')
        for m in models:
            if m not in t.index:
                continue
            y = ypos[m]
            mu, half = t.loc[m, 'AUC'], t.loc[m, 'AUCm']
            c = FAMILY_COLOR[FAMILY[m]]
            ax.plot([mu - half, mu + half], [y, y], color=c, lw=1.6,
                    solid_capstyle='round', alpha=.75, zorder=2)
            ax.plot([mu], [y], 'o', ms=5.5, color=c, mec='white', mew=1.0, zorder=3)
            ax.annotate(f'{mu:.3f}', (mu, y), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=6.2, color=MUTED)
        ax.set_yticks(range(len(models)))
        if j == 0:
            ax.set_yticklabels([SHORT.get(m, m) + ('†' if m in CORRECTED else '')
                                for m in models[::-1]], fontsize=8, color=INK)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(-0.75, len(models) - 0.25)
        ax.set_xlim(xlo, xhi)
        ax.set_xticks([0.70, 0.80, 0.90])
        ax.tick_params(axis='x', labelsize=7.5, colors=MUTED, length=2)
        ax.tick_params(axis='y', length=0)
        ax.set_title(title, fontsize=8.2, color=INK, pad=6, linespacing=1.35)
        ax.set_xlabel('ROC-AUC', fontsize=7.8, color=MUTED)
        ax.xaxis.grid(True, color=GRID, lw=.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        # family separators
        for b in (2.5, 6.5, 8.5, 9.5):
            ax.axhline(len(models) - 1 - b, color=GRID, lw=.6, ls=(0, (2, 2)))

    # ── Panel B: critical-difference diagram ────────────────────────────────
    axb = fig.add_subplot(gs[1, :])
    lo, hi = 1, k
    axb.set_xlim(hi + 3.0, lo - 3.0)          # rank 1 (best) on the right;
    # the wide margins are label gutters, not data range
    axb.set_ylim(-0.35, 2.5)
    axb.axis('off')
    axb.plot([lo, hi], [2.08, 2.08], color=INK, lw=1.0)
    for r in range(lo, hi + 1):
        axb.plot([r, r], [2.08, 2.20], color=INK, lw=1.0)
        axb.text(r, 2.30, str(r), ha='center', va='bottom', fontsize=7, color=MUTED)
    axb.text((lo + hi) / 2, 2.52, 'mean rank across all 20 split units '
             '(1 = best)', ha='center', va='bottom', fontsize=7.8, color=INK)

    order = list(mean_rank.index)
    half_n = int(np.ceil(len(order) / 2))
    for idx, m in enumerate(order):
        r = mean_rank[m]
        right = idx < half_n
        row = (half_n - 1 - idx) if right else (idx - half_n)
        ylev = 1.78 - 0.225 * row
        xend = lo - 0.40 if right else hi + 0.40
        c = FAMILY_COLOR[FAMILY[m]]
        axb.plot([r, r], [2.08, ylev], color=c, lw=1.0)
        axb.plot([r, xend], [ylev, ylev], color=c, lw=1.0)
        axb.plot([r], [2.08], 'o', ms=4, color=c, mec='white', mew=.8, zorder=4)
        axb.text(xend + (-0.12 if right else 0.12), ylev,
                 f'{SHORT.get(m, m)}{"†" if m in CORRECTED else ""}  {r:.1f}',
                 ha='left' if right else 'right', va='center', fontsize=7.2,
                 color=INK)

    # cliques: maximal groups whose rank spread <= CD
    yb = 0.42
    cliques = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and mean_rank[order[j + 1]] - mean_rank[order[i]] <= cd:
            j += 1
        if j > i:
            cliques.append((mean_rank[order[i]], mean_rank[order[j]]))
    uniq = list(dict.fromkeys(cliques))          # identical spans arise repeatedly
    merged = [c for c in uniq
              if not any(o[0] <= c[0] and c[1] <= o[1] and o != c for o in uniq)]
    for a, b in merged:
        axb.plot([a - 0.04, b + 0.04], [yb, yb], color=MUTED, lw=3.2,
                 solid_capstyle='round')
        yb -= 0.17
    axb.plot([lo, lo + cd], [-0.18, -0.18], color=INK, lw=1.2)  # CD scale bar
    for x in (lo, lo + cd):
        axb.plot([x, x], [-0.24, -0.12], color=INK, lw=1.2)
    axb.text(lo + cd / 2, -0.33, f'CD = {cd:.2f}', ha='center', va='top',
             fontsize=7.2, color=MUTED)

    handles = [Line2D([], [], color=v, lw=2.2, label=kk)
               for kk, v in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.6, 1.002), handlelength=1.5,
               columnspacing=1.3, handletextpad=.5)
    fig.text(0.013, 0.975, 'A', fontsize=11, fontweight='bold', color=INK)
    fig.text(0.013, 0.325, 'B', fontsize=11, fontweight='bold', color=INK)

    os.makedirs(f'{HERE}/figures', exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{HERE}/figures/mmp_main_results.{ext}', dpi=400,
                    facecolor='white')
    print(f'Friedman chi2={chi2:.1f}, p={p_fried:.3g}  (k={k} models, N={N} units)')
    if excluded:
        print(f'excluded from panel B (absent in >=1 protocol): '
              f'{[SHORT.get(m, m) for m in excluded]}')
    print(f'Nemenyi CD = {cd:.3f}\n')
    print('mean ranks (1 = best):')
    for m, r in mean_rank.items():
        print(f'  {SHORT.get(m, m):16s} {r:5.2f}  {FAMILY[m]}')
    print(f'\nnot-significantly-different groups: {len(merged)}')
    for a, b in merged:
        grp = [SHORT.get(m, m) for m in order if a <= mean_rank[m] <= b]
        print(f'  ranks {a:.2f}-{b:.2f}: {", ".join(grp)}')
    print(f'\n-> figures/mmp_main_results.pdf / .png')


if __name__ == '__main__':
    main()
