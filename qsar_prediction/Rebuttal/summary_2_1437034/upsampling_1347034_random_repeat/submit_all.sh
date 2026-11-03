#!/bin/bash
# AID 1347034 only (no AID 654), 5 repeated random 8:1:1 splits, up-sampled train.
# prep.slurm builds the splits; RF/SVM/XGB, the 4 GNNs and Graphormer wait on it.
# GraphGPS waits on prep_gt (Laplacian PE), MV-GNN waits on prep_mvgnn (bond graphs).
# consensus.slurm waits on all 10 models.
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

PREP_ID=$(sbatch --parsable prep.slurm)
echo "Submitted prep.slurm as job $PREP_ID"

MODEL_IDS=()
for job in rf svm xgb gcn gat gin sage graphormer; do
    JID=$(sbatch --parsable --dependency=afterok:$PREP_ID "${job}.slurm")
    echo "Submitted ${job}.slurm as job $JID (waits for $PREP_ID)"
    MODEL_IDS+=($JID)
done

PREP_GT_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID prep_gt.slurm)
echo "Submitted prep_gt.slurm as job $PREP_GT_ID"
GPS_ID=$(sbatch --parsable --dependency=afterok:$PREP_GT_ID graphgps.slurm)
echo "Submitted graphgps.slurm as job $GPS_ID (waits for $PREP_GT_ID)"
MODEL_IDS+=($GPS_ID)

PREP_MVGNN_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID prep_mvgnn.slurm)
echo "Submitted prep_mvgnn.slurm as job $PREP_MVGNN_ID"
MVGNN_ID=$(sbatch --parsable --dependency=afterok:$PREP_MVGNN_ID mvgnn.slurm)
echo "Submitted mvgnn.slurm as job $MVGNN_ID (waits for $PREP_MVGNN_ID)"
MODEL_IDS+=($MVGNN_ID)

DEP=$(IFS=:; echo "${MODEL_IDS[*]}")
CONSENSUS_ID=$(sbatch --parsable --dependency=afterok:$DEP consensus.slurm)
echo "Submitted consensus.slurm as job $CONSENSUS_ID (waits for all 10 models)"
echo "Check status with: squeue -u \$USER"
