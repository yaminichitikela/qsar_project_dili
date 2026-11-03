#!/bin/bash
# Submits the full InChIKey-cleaned, two-stage DILI MV-GNN 5-fold-CV pipeline:
#   prep -> {6 Stage-1 hyperparameter-search configs, each running its own
#            internal 5-fold CV} -> select_stage1_config (picks the winner
#            by mean CV val AUC) -> {5 Stage-1 fold jobs, retraining ONLY the
#            winning config per fold, this time saving checkpoints}
#         -> aggregate_stage1_cv (ensembles the 5 Stage-1 fold-models ->
#            Proxy-only report)
#         -> {5 Stage-2 fold jobs, each fine-tuning its own fold's Stage-1
#            checkpoint on clin_train_fold + NCV-888}
#         -> aggregate_stage2_cv (ensembles the 5 Stage-2 fold-models ->
#            Proxy+DILI report)
#
# Two real bugs fixed vs. the original DILI_mvgnn_cv: csmi-string dedup ->
# InChIKey dedup, and the clinical pool only excluding Gold-1111 (not
# held-out-223 or NCV-888) -> now excludes all three by InChIKey.
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

PREP_ID=$(sbatch --parsable prep_dili_cv.slurm)
echo "Submitted prep_dili_cv.slurm as job $PREP_ID"

CFG_IDS=()
for i in 0 1 2 3 4 5; do
    JID=$(sbatch --parsable --dependency=afterok:$PREP_ID "config_${i}.slurm")
    echo "Submitted config_${i}.slurm as job $JID (waits for $PREP_ID)"
    CFG_IDS+=($JID)
done

CFG_DEP=$(IFS=:; echo "${CFG_IDS[*]}")
SELECT_ID=$(sbatch --parsable --dependency=afterok:$CFG_DEP select_stage1_config.slurm)
echo "Submitted select_stage1_config.slurm as job $SELECT_ID (waits for all 6 configs: $CFG_DEP)"

S1_IDS=()
for i in 0 1 2 3 4; do
    JID=$(sbatch --parsable --dependency=afterok:$SELECT_ID "stage1_fold_${i}.slurm")
    echo "Submitted stage1_fold_${i}.slurm as job $JID (waits for $SELECT_ID)"
    S1_IDS+=($JID)
done

S1_DEP=$(IFS=:; echo "${S1_IDS[*]}")
AGG1_ID=$(sbatch --parsable --dependency=afterok:$S1_DEP aggregate_stage1_cv.slurm)
echo "Submitted aggregate_stage1_cv.slurm as job $AGG1_ID (waits for all 5 Stage-1 folds: $S1_DEP)"

S2_IDS=()
for i in 0 1 2 3 4; do
    # each Stage-2 fold only needs ITS OWN Stage-1 fold checkpoint, but
    # gating all 5 on aggregate_stage1_cv keeps the dependency chain simple
    # and guarantees every checkpoint is confirmed present before Stage-2 starts.
    JID=$(sbatch --parsable --dependency=afterok:$AGG1_ID "stage2_fold_${i}.slurm")
    echo "Submitted stage2_fold_${i}.slurm as job $JID (waits for $AGG1_ID)"
    S2_IDS+=($JID)
done

S2_DEP=$(IFS=:; echo "${S2_IDS[*]}")
AGG2_ID=$(sbatch --parsable --dependency=afterok:$S2_DEP aggregate_stage2_cv.slurm)
echo "Submitted aggregate_stage2_cv.slurm as job $AGG2_ID (waits for all 5 Stage-2 folds: $S2_DEP)"

echo
echo "All DILI MV-GNN CV jobs submitted (1 prep + 6 configs + 1 select + 5 stage1 folds"
echo "+ 1 aggregate1 + 5 stage2 folds + 1 aggregate2 = 20 jobs)."
echo "Check status with: squeue -u \$USER"
