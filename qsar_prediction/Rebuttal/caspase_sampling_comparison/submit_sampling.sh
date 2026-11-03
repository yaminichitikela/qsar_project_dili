#!/bin/bash
# Submits the caspase-3/7 sampling-strategy comparison (Reviewer 2's
# downsampling justification + Reviewer 1 #13's SMOTE ask) for BOTH
# datasets (HepG2 = summary_2_1437034, CHO-K1 = summary_1347037) from this
# one self-contained folder. Does not touch wahab_jobs_s2 / wahab_jobs_s3 --
# those stay untouched reference copies of the original pipeline.
#
# Assumes the 4 variant data folders already exist under
# ./data/{hepg2,cho_k1}/{v1_ratio_1to5,v2_similarity,v3_diversity,v4_smote}/
# (built locally by v1..v4_*.py in this same folder -- CPU-only, already run).
#
# Per (dataset, variant):
#   v1_ratio_1to5, v2_similarity, v3_diversity  -> classical (RF/SVM/XGB) +
#     GCN/GAT/GIN/SAGE/Graphormer directly, GraphGPS/MV-GNN after their own
#     prep job (Laplacian PE / bond-graphs) for that (dataset, variant).
#   v4_smote -> classical (RF/SVM/XGB) ONLY -- no graphs exist for this
#     variant (SMOTE synthesizes fingerprint vectors, not valid molecules).
set -e
cd "$(dirname "$0")/jobs_sampling"
mkdir -p ../logs

DATASETS=(hepg2 cho_k1)
GRAPH_VARIANTS=(v1_ratio_1to5 v2_similarity v3_diversity)
ALL_VARIANTS=(v1_ratio_1to5 v2_similarity v3_diversity v4_smote)

ALL_IDS=()

for ds in "${DATASETS[@]}"; do
    for v in "${ALL_VARIANTS[@]}"; do
        echo "=== ${ds} / ${v} ==="
        for job in rf svm xgb; do
            JID=$(sbatch --parsable --job-name="${job}_${ds}_${v}" "${job}_sampling.slurm" "$ds" "$v")
            echo "  ${job}_sampling.slurm $ds $v -> job $JID"
            ALL_IDS+=($JID)
        done
    done

    for v in "${GRAPH_VARIANTS[@]}"; do
        for job in gcn gat gin sage graphormer; do
            JID=$(sbatch --parsable --job-name="${job}_${ds}_${v}" "${job}_sampling.slurm" "$ds" "$v")
            echo "  ${job}_sampling.slurm $ds $v -> job $JID"
            ALL_IDS+=($JID)
        done

        PREP_GT_ID=$(sbatch --parsable --job-name="prep_gt_${ds}_${v}" prep_gt_sampling.slurm "$ds" "$v")
        echo "  prep_gt_sampling.slurm $ds $v -> job $PREP_GT_ID"
        GPS_ID=$(sbatch --parsable --job-name="graphgps_${ds}_${v}" --dependency=afterok:$PREP_GT_ID graphgps_sampling.slurm "$ds" "$v")
        echo "  graphgps_sampling.slurm $ds $v -> job $GPS_ID (waits for $PREP_GT_ID)"
        ALL_IDS+=($GPS_ID)

        PREP_MVGNN_ID=$(sbatch --parsable --job-name="prep_mvgnn_${ds}_${v}" prep_mvgnn_sampling.slurm "$ds" "$v")
        echo "  prep_mvgnn_sampling.slurm $ds $v -> job $PREP_MVGNN_ID"
        MVGNN_ID=$(sbatch --parsable --job-name="mvgnn_${ds}_${v}" --dependency=afterok:$PREP_MVGNN_ID mvgnn_sampling.slurm "$ds" "$v")
        echo "  mvgnn_sampling.slurm $ds $v -> job $MVGNN_ID (waits for $PREP_MVGNN_ID)"
        ALL_IDS+=($MVGNN_ID)
    done
done

echo
echo "All jobs submitted: ${#ALL_IDS[@]} total"
echo "(2 datasets x [3 variants x 10 models + 1 variant x 3 classical-only] = 78 model jobs, plus 12 prep jobs)"
echo "Check status with: squeue -u \$USER"
