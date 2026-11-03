#!/bin/bash
# MMP (AID 720637), up-sampled training set, three protocols.
#   ./submit_all.sh                     -> all three protocols
#   ./submit_all.sh loo scaffold        -> only the named ones
# Per protocol: prep builds the splits; RF/SVM/XGB, the 4 GNNs and Graphormer
# wait on it; GraphGPS waits on prep_gt (Laplacian PE); MV-GNN waits on
# prep_mvgnn (bond graphs); consensus waits on all 10 models.
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs
PROTOCOLS=("$@"); [ ${#PROTOCOLS[@]} -eq 0 ] && PROTOCOLS=(random_repeat loo scaffold)

for P in "${PROTOCOLS[@]}"; do
    case "$P" in random_repeat|loo|scaffold) ;; *) echo "unknown protocol: $P"; exit 1;; esac
    echo "=== $P"
    PREP_ID=$(sbatch --parsable prep_$P.slurm)
    echo "  prep_$P -> $PREP_ID"
    MODEL_IDS=()
    for job in rf svm xgb gcn gat gin sage graphormer; do
        JID=$(sbatch --parsable --dependency=afterok:$PREP_ID ${job}_$P.slurm)
        echo "  ${job}_$P -> $JID"
        MODEL_IDS+=($JID)
    done
    GT_PREP=$(sbatch --parsable --dependency=afterok:$PREP_ID prep_gt_$P.slurm)
    JID=$(sbatch --parsable --dependency=afterok:$GT_PREP graphgps_$P.slurm)
    echo "  prep_gt_$P -> $GT_PREP, graphgps_$P -> $JID"
    MODEL_IDS+=($JID)
    MV_PREP=$(sbatch --parsable --dependency=afterok:$PREP_ID prep_mvgnn_$P.slurm)
    JID=$(sbatch --parsable --dependency=afterok:$MV_PREP mvgnn_$P.slurm)
    echo "  prep_mvgnn_$P -> $MV_PREP, mvgnn_$P -> $JID"
    MODEL_IDS+=($JID)
    DEP=$(IFS=:; echo "${MODEL_IDS[*]}")
    JID=$(sbatch --parsable --dependency=afterok:$DEP consensus_$P.slurm)
    echo "  consensus_$P -> $JID (waits for all 10 models)"
done
echo "Check status with: squeue -u \$USER"
