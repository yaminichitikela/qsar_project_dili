# TX2017-Structure under the three split protocols (up-sampled)

Runs the TX2017-Structure pipeline (Xu et al. 2017, as replicated in
[`../DILI_tx2017`](../DILI_tx2017)) under the rebuttal split protocols, with
the training set balanced by random over-sampling. Structure-only: ECFP4 and
nothing else. Design notes are in `common_tx.py`.

| Protocol | Units | Training set |
|---|---|---|
| `random_repeat` | 5 | random 80/10/10, 1:1 up-sampled |
| `cv` | 10 | 10-fold, fold i internal test, fold i+1 val, 1:1 up-sampled |
| `scaffold` | 5 | Murcko 8:1:1 (Yang et al.), 1:1 up-sampled |

The split indices in `splits/` are copied unchanged from
`../DILI_split_protocols`, so every unit holds exactly the same compounds as
the graph and classical DILI runs and the tables line up row for row.

Kept from TX2017: ECFP4 at 1024 bits, the five classifiers (Naive Bayes, SVM,
Random Forest, neural network, XGBoost) with their original settings,
imblearn's `RandomOverSampler`, and the winning feature-selection setting from
the original search (Random-Forest Gini importance, top 180 bits). SVM with
over-sampling was that search's winning combination for both stages.

Changed, and why: the 20 random train/validation reshuffles are replaced by the
three protocols' units, so TX2017 sits on the same splits as everything else;
and the 180 bits are re-selected inside each unit from its own training rows,
so validation never leaks into feature selection. Selection runs on the raw
training rows before over-sampling, as in the original.

Up-sampling is applied to the training rows only. Validation and the internal
test split keep their natural class ratio, and Held-223 is never resampled.

## Run

Everything runs on a laptop in roughly 40 minutes; no cluster needed. SVM is
the slow step at about 50 seconds per unit.

```bash
cd DILI_tx2017_splits
python3 prepare_tx_features.py      # once: builds the 1024-bit fingerprints
./run_all_local.sh                  # 20 units x 2 stages; finished units are skipped
python3 aggregate_tx.py             # summary + LaTeX tables (add 0.8 for that cutoff)
```

A single unit, for checking: `python3 train_tx_unit.py scaffold 2 0`.
Set `TX_SMOKE=1` for a fast wiring test, or `TX_CLFS=NB,RF` to restrict the
classifier list.

## Outputs (`results/`)

- `tx_per_unit_all.csv`: every unit, stage, classifier and split.
- `tx_summary.csv`: mean and SD per protocol, stage, classifier and cutoff, for Held-223 and the internal test split.
- `tables_tx2017_splits_upsampled_cpt50.tex`: three LaTeX tables, one per split, Stage 1 and Stage 2 stacked, in the same format as the other rebuttal tables.
- `<protocol>/feat_s<stage>_u<unit>.npy`: the bits selected inside each unit.
