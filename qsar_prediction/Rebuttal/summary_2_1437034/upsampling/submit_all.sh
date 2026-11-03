#!/bin/bash
# Submits prep.slurm first, then all 7 base training jobs (RF, SVM, XGBoost,
# GCN, GAT, GIN, SAGE) + Graphormer with --dependency=afterok on the prep job
# (all 8 only need the plain split graphs). GraphGPS and MV-GNN^cross each
# additionally need their own prep job (Laplacian PE / bond-graphs
# respectively), which itself waits on prep.slurm. Finally consensus.slurm
# waits on ALL 10 models (including MV-GNN) -- compute_consensus.py now
# always loads mvgnn_test_probs.npy too (to compute Consensus-MVGNN), so it
# must not start before mvgnn.slurm finishes. Everything that can run in
# parallel does.
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
echo "Submitted prep_gt.slurm as job $PREP_GT_ID (waits for $PREP_ID)"

GPS_ID=$(sbatch --parsable --dependency=afterok:$PREP_GT_ID graphgps.slurm)
echo "Submitted graphgps.slurm as job $GPS_ID (waits for $PREP_GT_ID)"
MODEL_IDS+=($GPS_ID)

PREP_MVGNN_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID prep_mvgnn.slurm)
echo "Submitted prep_mvgnn.slurm as job $PREP_MVGNN_ID (waits for $PREP_ID)"

MVGNN_ID=$(sbatch --parsable --dependency=afterok:$PREP_MVGNN_ID mvgnn.slurm)
echo "Submitted mvgnn.slurm as job $MVGNN_ID (waits for $PREP_MVGNN_ID)"
MODEL_IDS+=($MVGNN_ID)

DEP=$(IFS=:; echo "${MODEL_IDS[*]}")
CONSENSUS_ID=$(sbatch --parsable --dependency=afterok:$DEP consensus.slurm)
echo "Submitted consensus.slurm as job $CONSENSUS_ID (waits for all 10 models: $DEP)"

echo
echo "All jobs submitted (7 base + Graphormer + GraphGPS + MV-GNN + Consensus)."
echo "Check status with: squeue -u \$USER"
