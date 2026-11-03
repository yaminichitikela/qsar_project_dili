#!/bin/bash
# Trains the corrected graph transformers for the SubgraphX rerun.
#
# Target: the repeated SCAFFOLD split, repeat 0, Stage 1 (Proxy-only) -- the
# setting the manuscript reports as hardest and the one where Graphormer is the
# best model of all fifteen (ROC-AUC 0.622, PR-AUC 0.786). Explaining a model
# where it leads is the defensible pairing; the random-split Stage-2 run is
# where Graphormer is weakest (0.591), so it is not used here.
#
# Stage 1 needs no warm start, so only one stage per architecture is trained.
# Resumable: a stage whose weights already exist is skipped.
set -u
cd "$(dirname "$0")"
export DILI_SAVE_STAGE2=1
PROTO=scaffold_up
for fam in Graphormer GraphGPS; do
  w="results/${PROTO}/states/${fam}_s1_u0.pt"
  if [ -f "$w" ]; then echo "skip $fam (have $w)"; continue; fi
  echo "=== $fam stage 1 on $PROTO  $(date +%H:%M)"
  python3 train_protocol.py "$PROTO" "$fam" 1 0 || echo "FAILED $fam"
done
echo "XAI MODELS DONE"
