#!/bin/bash
# Submits prep_loo.slurm first (builds the 10 folds), then all 7 base LOO
# training jobs (RF, SVM, XGBoost, GCN, GAT, GIN, SAGE) + Graphormer_loo with
# --dependency=afterok on the prep job. GraphGPS_loo additionally needs
# prep_gt_loo.slurm (per-fold Laplacian PE), which itself waits on
# prep_loo.slurm. Finally consensus_loo.slurm waits on all 9 individual model
# jobs. Everything that can run in parallel does.
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

PREP_ID=$(sbatch --parsable prep_loo.slurm)
echo "Submitted prep_loo.slurm as job $PREP_ID"

MODEL_IDS=()
for job in rf_loo svm_loo xgb_loo gcn_loo gat_loo gin_loo sage_loo graphormer_loo; do
    JID=$(sbatch --parsable --dependency=afterok:$PREP_ID "${job}.slurm")
    echo "Submitted ${job}.slurm as job $JID (waits for $PREP_ID)"
    MODEL_IDS+=($JID)
done

PREP_GT_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID prep_gt_loo.slurm)
echo "Submitted prep_gt_loo.slurm as job $PREP_GT_ID (waits for $PREP_ID)"

GPS_ID=$(sbatch --parsable --dependency=afterok:$PREP_GT_ID graphgps_loo.slurm)
echo "Submitted graphgps_loo.slurm as job $GPS_ID (waits for $PREP_GT_ID)"
MODEL_IDS+=($GPS_ID)

PREP_MVGNN_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID prep_mvgnn_loo.slurm)
echo "Submitted prep_mvgnn_loo.slurm as job $PREP_MVGNN_ID (waits for $PREP_ID)"

MVGNN_ID=$(sbatch --parsable --dependency=afterok:$PREP_MVGNN_ID mvgnn_loo.slurm)
echo "Submitted mvgnn_loo.slurm as job $MVGNN_ID (waits for $PREP_MVGNN_ID)"
MODEL_IDS+=($MVGNN_ID)

DEP=$(IFS=:; echo "${MODEL_IDS[*]}")
CONSENSUS_ID=$(sbatch --parsable --dependency=afterok:$DEP consensus_loo.slurm)
echo "Submitted consensus_loo.slurm as job $CONSENSUS_ID (waits for all 10 models: $DEP)"

echo
echo "All LOO jobs submitted (7 base + Graphormer + GraphGPS + MV-GNN + Consensus)."
echo "Check status with: squeue -u \$USER"
