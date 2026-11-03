# Revision: results and code

This folder holds the revised manuscript and every analysis added or re-run for
the revision.

**Revised manuscript:** [`main_paper.tex`](main_paper.tex) (bibliography in
[`reference.bib`](reference.bib)).
**All result tables in one workbook:**
[`QSAR_Rebuttal_All_Results.xlsx`](QSAR_Rebuttal_All_Results.xlsx).

## Where each result lives

### In-vitro assay models (summary assays)

| Endpoint | AID | Folder |
|---|---|---|
| Mitochondrial membrane potential (MMP) | 720637 | [`Summary_1_models/`](Summary_1_models) |
| Caspase-3/7, CHO-K1 | 1347037 | [`summary_1347037/`](summary_1347037) |
| Caspase-3/7, HepG2 | 1347034 | [`summary_2_1437034/`](summary_2_1437034) |
| Raw assay tables | 720637, 1347034, 1347037 | [`summary datasets/`](summary%20datasets) |

Inside each one, `cleaning/` covers deduplication, `modeling/` the baseline
models, `upsampling/` the 1:1 random up-sampling runs, and
`upsampling_protocols/` the split protocols (10-fold CV, 5× random repeat,
Murcko scaffold).

Related analyses:

- [`summary_1_assay_graph_optuna/`](summary_1_assay_graph_optuna): Optuna
  hyperparameter search for the GNN and graph-transformer models (LOO, repeated
  and scaffold splits).
- [`graph_transformer_corrected/`](graph_transformer_corrected): Graphormer and
  GraphGPS re-run with the corrected implementations, plus consensus.
- [`caspase_sampling_comparison/`](caspase_sampling_comparison): comparison of
  class-balancing strategies on Caspase-3/7.
- [`repeated_random_split/`](repeated_random_split),
  [`random_repeat_results/`](random_repeat_results),
  [`wahab_random_repeat/`](wahab_random_repeat): repeated random-split runs
  for CHO-K1, HepG2 and MMP.
- [`AD_cutoff_caspase.ipynb`](AD_cutoff_caspase.ipynb): applicability-domain
  cutoff analysis.

### DILI prediction

- [`DILI_split_protocols/`](DILI_split_protocols): all ten model families and
  the consensus ensembles under random-repeat, 10-fold CV and scaffold splits,
  with down-sampling and up-sampling, scored on Held-223. See its
  [README](DILI_split_protocols/README.md).
- [`DILI_assay_link/`](DILI_assay_link): links the in-vitro assay predictions
  to DILI (DILI rate by caspase/MMP pattern, leakage audit).
- [`DILI_tx2017/`](DILI_tx2017), [`DILI_tx2017_splits/`](DILI_tx2017_splits),
  [`DILI_tx2017_2048/`](DILI_tx2017_2048),
  [`DILI_tx2017_classical_hp/`](DILI_tx2017_classical_hp),
  [`DILI_tx2017_code/`](DILI_tx2017_code): replication of the Xu et al. (2017)
  structure-only baseline, then the same pipeline on our splits, at 2048 bits
  and with tuned hyperparameters. See its
  [README](DILI_tx2017_splits/README.md).
- [`DILI_mvgnn/`](DILI_mvgnn), [`DILI_mvgnn_cv/`](DILI_mvgnn_cv),
  [`DILI_mvgnn_proxy/`](DILI_mvgnn_proxy),
  [`DILI_proxy_clean/`](DILI_proxy_clean): MV-GNN and proxy-assay DILI
  models.

### Structural analysis

- [`structural activation/`](structural%20activation): BRICS fragment
  enrichment (Fisher test, Benjamini–Hochberg at α = 0.10).
- [`updated_figures/`](updated_figures): Figures 8–10 rebuilt on the summary
  assays. See its [README](updated_figures/README.md).
- [`ncats_13_cell_types/`](ncats_13_cell_types): the NCATS cell-line panels
  used in the cross-cell-line comparison.

### Cross-cutting results

- [`results/`](results): deduplicated DILI sets (NCV-888, Held-223,
  Gold-1111), significance tests, confidence-interval summaries and assay
  statistics.
- [`figures/`](figures): the manuscript figures held in this folder, with the
  data behind them (`*_values.csv`). [`figure_data/`](figure_data) holds the
  scripts that assemble that data.
- Top-level `*.py`: scripts that build the figures, confidence intervals and
  significance tests from the result CSVs. They resolve paths relative to this
  folder.

## Not included

Model checkpoints and cached feature arrays (`*.pt`, `*.npy`, about 20 GB) and
HPC job logs are left out. The `cleaning/`, `prepare_*` and `jobs/` scripts in
each folder regenerate them. Every reported metric is kept as CSV.
