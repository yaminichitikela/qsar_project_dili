#!/bin/bash
# Trains the four GNNs for the Consensus-GNN SubgraphX run.
#
# Target: 10-fold CV with 1:1 downsampled training, fold 9, Stage 2
# (Proxy+DILI) -- the setting the manuscript reports for Consensus-GNN
# (ROC-AUC 0.6315 +/- 0.0241 over the ten folds). Fold 9's own Consensus-GNN
# ROC-AUC (0.6202) is the closest of the ten to that mean, so the explained
# model is representative of the reported number rather than the most
# flattering fold (fold 1, 0.6613).
#
# Stage 2 warm-starts from Stage 1, so both stages are trained per family and
# both sets of weights are kept (DILI_SAVE_STAGE2=1 -- Stage-2 weights are
# normally discarded).
# Resumable: a stage whose weights already exist is skipped.
set -u
cd "$(dirname "$0")"
export DILI_SAVE_STAGE2=1
PROTO=cv
UNIT=9
for fam in GCN GAT GIN SAGE; do
  for stage in 1 2; do
    w="results/${PROTO}/states/${fam}_s${stage}_u${UNIT}.pt"
    if [ -f "$w" ]; then echo "skip ${fam} stage ${stage} (have $w)"; continue; fi
    echo "=== ${fam} stage ${stage} on ${PROTO} unit ${UNIT}  $(date +%H:%M)"
    python3 train_protocol.py "$PROTO" "$fam" "$stage" "$UNIT" || echo "FAILED ${fam} s${stage}"
  done
done
echo "XAI GNN MODELS DONE"
