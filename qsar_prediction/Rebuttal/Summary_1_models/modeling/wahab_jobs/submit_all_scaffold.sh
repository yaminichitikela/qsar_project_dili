#!/bin/bash
# Submits prep_scaffold.slurm first (builds the 5 repeated scaffold splits),
# then all 7 base scaffold training jobs (RF, SVM, XGBoost, GCN, GAT, GIN,
# SAGE) + Graphormer_scaffold with --dependency=afterok on the prep job.
# GraphGPS_scaffold additionally needs prep_gt_scaffold.slurm (per-repeat
# Laplacian PE), which itself waits on prep_scaffold.slurm. Finally
# consensus_scaffold.slurm waits on all 9 individual model jobs. Everything
# that can run in parallel does.
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

PREP_ID=$(sbatch --parsable prep_scaffold.slurm)
echo "Submitted prep_scaffold.slurm as job $PREP_ID"

MODEL_IDS=()
for job in rf_scaffold svm_scaffold xgb_scaffold gcn_scaffold gat_scaffold gin_scaffold sage_scaffold graphormer_scaffold; do
    JID=$(sbatch --parsable --dependency=afterok:$PREP_ID "${job}.slurm")
    echo "Submitted ${job}.slurm as job $JID (waits for $PREP_ID)"
    MODEL_IDS+=($JID)
done

PREP_GT_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID prep_gt_scaffold.slurm)
echo "Submitted prep_gt_scaffold.slurm as job $PREP_GT_ID (waits for $PREP_ID)"

GPS_ID=$(sbatch --parsable --dependency=afterok:$PREP_GT_ID graphgps_scaffold.slurm)
echo "Submitted graphgps_scaffold.slurm as job $GPS_ID (waits for $PREP_GT_ID)"
MODEL_IDS+=($GPS_ID)

PREP_MVGNN_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID prep_mvgnn_scaffold.slurm)
echo "Submitted prep_mvgnn_scaffold.slurm as job $PREP_MVGNN_ID (waits for $PREP_ID)"

MVGNN_ID=$(sbatch --parsable --dependency=afterok:$PREP_MVGNN_ID mvgnn_scaffold.slurm)
echo "Submitted mvgnn_scaffold.slurm as job $MVGNN_ID (waits for $PREP_MVGNN_ID)"
MODEL_IDS+=($MVGNN_ID)

DEP=$(IFS=:; echo "${MODEL_IDS[*]}")
CONSENSUS_ID=$(sbatch --parsable --dependency=afterok:$DEP consensus_scaffold.slurm)
echo "Submitted consensus_scaffold.slurm as job $CONSENSUS_ID (waits for all 10 models: $DEP)"

echo
echo "All scaffold-split jobs submitted (7 base + Graphormer + GraphGPS + MV-GNN + Consensus)."
echo "Check status with: squeue -u \$USER"
