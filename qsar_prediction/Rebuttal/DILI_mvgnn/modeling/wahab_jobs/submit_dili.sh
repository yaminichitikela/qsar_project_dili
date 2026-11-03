#!/bin/bash
# Submits prep_dili.slurm first (builds clin_train/clin_val/held-out-223
# graphs), then mvgnn.slurm with --dependency=afterok, so training only
# starts once the split data exists.
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

PREP_ID=$(sbatch --parsable prep_dili.slurm)
echo "Submitted prep_dili.slurm as job $PREP_ID"

MVGNN_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID mvgnn.slurm)
echo "Submitted mvgnn.slurm as job $MVGNN_ID (waits for $PREP_ID)"

echo
echo "DILI MV-GNN^cross jobs submitted. Check status with: squeue -u \$USER"
