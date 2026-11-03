# Updated cross-cell-line figures (summary assays)

Rebuild of Figures 8, 9 and 10 on the **summary** assays, replacing the raw
assays that were removed from Table 1 and Table 3.

| | published figures used | this rebuild uses |
|---|---|---|
| HepG2 | `combined_hepg2_casepase.csv` = AID 654 + raw **1346978**, 241 actives | **AID 1347034** summary, 303 actives |
| CHO-K1 | raw **AID 1346980**, 115 actives | **AID 1347037** summary, 171 actives |

The twelve NCATS panels (AID 655–667) are unchanged.

Everything else is transcribed from `pairwise_brics_comparison.ipynb` and
`split_top10_figure.py`: same BRICS decomposition, same one-sided Fisher test
with Benjamini–Hochberg at α = 0.10, same ≥10-actives rule, same MCS
highlighting and Morgan(2, 2048) fingerprints.

## Validation

Running the pipeline on the **old** sources reproduces the published statistics
exactly — epoxide OR 35.6 / p_adj 0.051, tetramethylcyclohexene ∞ / 0.001,
acetaldehyde 35.6 / 0.032, 1,1-dichloroethane ∞ / 0.011, chlorobenzene
10.7 / 0.022. So differences below are real, not artefacts of reimplementation.

## What changes

**Figure 8 — essentially unchanged.** Same ten compounds, same order, both MCS
pairs intact. Only CID 5284469 moves, 6 → 5 active cell lines (it is active in
the raw merged HepG2 set but not in AID 1347034). The caption stands as written.

**Figure 9 — unchanged** apart from being redrawn (same ten compounds).

**Figure 10 — all five published fragments survive, in the same cell lines.**
Two new ones appear.

| Fragment | Cell line | OR old | p_adj old | OR new | p_adj new |
|---|---|---|---|---|---|
| 1,2,3,3-Tetramethylcyclohexene | H-4-II-E | ∞ | 0.0012 | 73.5 | 0.0046 |
| Acetaldehyde | H-4-II-E | 35.6 | 0.0321 | 46.1 | 0.0229 |
| 1,1-Dichloroethane | HEK293 | ∞ | 0.0105 | ∞ | 0.0129 |
| Chlorobenzene | HEK293 | 10.7 | 0.0224 | 6.3 | **0.0626** |
| Epoxide | SK-N-SH | 35.6 | **0.0505** | 46.1 | 0.0315 |
| Acryloyl (vinyl ketone) | CHO-K1 | — | — | 16.5 | 0.0312 |
| 2-Chloroethyl | Jurkat | — | — | 24.2 | 0.0878 |

The epoxide now clears α = 0.05 and chlorobenzene no longer does, so the
manuscript's "four of the five enriched fragments also satisfy the conventional
α = 0.05" stays true — a different one fails.

## Manuscript text that must change

p.23: *"HepG2 (n = 241), CHO-K1 (n = 115), Jurkat (n = 48), and SH-SY5Y (n = 10)
returned zero significant fragments"* — under the summary assays HepG2 is
n = 303 (still zero) and SH-SY5Y n = 10 (still zero), but **CHO-K1 (n = 171) and
Jurkat (n = 48) each now return one fragment**.

Also note the published sentence names eight cell lines but accounts for only
seven: **NIH 3T3 (n = 12)** is omitted. It returns 4 significant fragments in
both the old and new runs, none of them unique to it.

## Files

| File | What |
|---|---|
| `rebuild_celllines.py` | main script — builds both matrices, runs enrichment, draws Figs 8/9 |
| `unique_frags.py` | filters to fragments significant in exactly one cell line |
| `make_fig10.py` | draws Figure 10 and the old/new comparison table |
| `fig8_top10_structures_summary.png` | Figure 8, summary assays |
| `fig9_tanimoto_summary.png` | Figure 9, summary assays |
| `fig10_cellline_fragments_summary.png` | Figure 10, summary assays |
| `celllines_old_vs_new.csv` | per-cell-line active counts and fragment counts |
| `fig10_old_vs_new.csv` | per-fragment OR and p_adj, old vs new |
| `unique_fragments_{old,new}.csv` | cell-line-unique fragments |
| `fragments_{old,new}_<cellline>.csv` | full enrichment output per cell line |

Reproduce with:

```
python rebuild_celllines.py && python unique_frags.py && python make_fig10.py
```

## Not covered here

The compound population active in ≥1 cell line grows **323 → 412** with the
summary assays. That affects only these figures. The caspase × MMP analysis
(Table 7, Figure 7, Table 8) uses the merged `AID_654_1347034` set and is
untouched by this rebuild.
