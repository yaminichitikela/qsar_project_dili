#!/bin/bash
# Phase 2 of the two-stage TX2017-Structure pipeline: per-setting jobs for
# both stages, then aggregate_results.py per stage. Requires Phase 1
# (prep + feature_selection.py 1/2) to have already finished, and
# gen_setting_jobs.py to have generated this stage's real setting_s<N>_*.slurm
# files (counts below are what gen_setting_jobs.py found: stage1=31,
# stage2=31).
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

S1_IDS=()
for i in $(seq 0 30); do
    JID=$(sbatch --parsable "setting_s1_${i}.slurm")
    echo "Submitted setting_s1_${i}.slurm as job $JID"
    S1_IDS+=($JID)
done
DEP1=$(IFS=:; echo "${S1_IDS[*]}")
AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP1 aggregate_stage1.slurm)
echo "Submitted aggregate_stage1.slurm as job $AGG1_ID (waits for 31 stage-1 settings)"

S2_IDS=()
for i in $(seq 0 30); do
    JID=$(sbatch --parsable "setting_s2_${i}.slurm")
    echo "Submitted setting_s2_${i}.slurm as job $JID"
    S2_IDS+=($JID)
done
DEP2=$(IFS=:; echo "${S2_IDS[*]}")
AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP2 aggregate_stage2.slurm)
echo "Submitted aggregate_stage2.slurm as job $AGG2_ID (waits for 31 stage-2 settings)"

echo
echo "Phase 2 submitted (31 + 31 settings + 2 aggregate = 64 jobs)."
echo "Check status with: squeue -u \$USER"
