# DILI split protocols

Applies the assay rebuttal split protocols to DILI, for all 10 model families
(RF, SVM, XGB, GCN, GAT, GIN, SAGE, MV-GNN, Graphormer, GraphGPS) and the five
consensus ensembles, in both stages. Held-223 is the fixed test set and is used
for scoring only. Full design notes are in `common.py`.

| Protocol | Units | Training set |
|---|---|---|
| `random_repeat` | 5 | random 80/10/10, 1:1 downsampled |
| `random_repeat_nodown` | 5 | same splits, natural ratio (downsampling control) |
| `cv` | 10 | 10-fold, fold i test, fold i+1 val, 1:1 downsampled |
| `scaffold` | 5 | Murcko 8:1:1 (Yang et al.), 1:1 downsampled |
| `random_repeat_up` | 5 | same splits as `random_repeat`, 1:1 **up-sampled** |
| `cv_up` | 10 | same folds as `cv`, 1:1 **up-sampled** |
| `scaffold_up` | 5 | same splits as `scaffold`, 1:1 **up-sampled** |

Up-sampling duplicates DILI-positive compounds with replacement up to the
negative count, keeping every negative, as in the assay up-sampling runs
(`summary_1347037/upsampling`, `.../upsampling_protocols`). It is applied to
the **training set only**: validation and the internal test set keep their
natural class ratio, and Held-223 is never resampled. Each `*_up` protocol
reuses its downsampled counterpart's splits, so the two are paired per unit.

Fixed against the current DILI table: model code (byte-identical copy of
`DILI_assay_link`, corrected graph transformers), hyperparameters
(`configs/`, the Structure-only winning configs), data arrays
(`DILI_assay_link/data_dili_link`), epochs and early stopping.

## Run on Wahab

The folder is self-contained: `data/` holds an unchanged copy of the DILI
arrays, so it is the only thing to upload (about 500 MB). From any directory on
the laptop:

```bash
rsync -av --exclude logs/ --exclude results/ --exclude __pycache__/ \
  "/Users/yamini/Desktop/projects/qsar casepase/qsar_prediction/Rebuttal/DILI_split_protocols" \
  ychit001@wahab.hpc.odu.edu:~/
```

On Wahab:

```bash
cd ~/DILI_split_protocols
mkdir -p logs results
./submit_all.sh --dry-run     # checks the data, lists what is missing
./submit_all.sh               # submits only what has no results yet
```

`submit_all.sh` skips any protocol/family/stage whose per-unit metric files
are already present, so re-running it after the downsampled protocols
finished submits just the three up-sampled protocols (400 tasks). Pass
`--force` to resubmit everything, or `--protocols "cv_up scaffold_up"` to pick
a subset.

Back on the laptop (model weights in `states/` are left on Wahab):

```bash
rsync -av --exclude 'states/' ychit001@wahab.hpc.odu.edu:~/DILI_split_protocols/results/ \
  "/Users/yamini/Desktop/projects/qsar casepase/qsar_prediction/Rebuttal/DILI_split_protocols/results/"
```

## Outputs (`results/`)

- `summary.csv`: mean and SD per protocol, stage, model and cutoff, for Held-223 (`test`) and the internal test (`itest`).
- `comparison_vs_baseline.csv`: change against the current DILI table, with Welch and Mann-Whitney p-values.
- `downsampling_effect.csv`: downsampled vs not, paired on the same 5 splits.
- `upsampling_effect.csv`: up-sampled vs 1:1 downsampled training, paired per unit on the shared splits, for all three split families.
- `cv_fold_ensemble.csv`: the 10 fold models averaged, per family.
- `tables_dili_split_protocols.tex`: LaTeX tables in the same format as `../tables_dili_repeated_split.tex`.
