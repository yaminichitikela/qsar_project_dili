#!/bin/bash
# Phase 1 of the two-stage TX2017-Structure pipeline: prep (InChIKey-cleaned
# ECFP4 features) -> feature_selection.py 1 (Proxy-only) and
# feature_selection.py 2 (Proxy+DILI), both depending only on prep since
# Stage-2 feature selection re-searches independently on clin_train+NCV-888
# (classical models have no weights to fine-tune, same reasoning as
# DILI_proxy_clean's classical Stage-2).
#
# The number of VALID (non-empty) feature-selection settings per stage is
# only known after this phase finishes (some AUC-ROC cutoffs may select 0
# features -- real single-stage precedent: 35/37 nominal settings were valid,
# the two highest AUC cutoffs 0.56/0.57 selected 0 features). Do not
# generate/submit the per-setting jobs until you've pulled
# results/stage1/feature_selection.json and results/stage2/feature_selection.json
# back from Wahab and confirmed the real counts -- see gen_setting_jobs.py.
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

PREP_ID=$(sbatch --parsable prep.slurm)
echo "Submitted prep.slurm as job $PREP_ID"

FS1_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID feature_selection_stage1.slurm)
echo "Submitted feature_selection_stage1.slurm as job $FS1_ID (waits for $PREP_ID)"

FS2_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID feature_selection_stage2.slurm)
echo "Submitted feature_selection_stage2.slurm as job $FS2_ID (waits for $PREP_ID)"

echo
echo "Phase 1 submitted (prep + feature_selection x2 = 3 jobs). Check status with: squeue -u \$USER"
echo "Once both feature_selection jobs finish, pull back:"
echo "  results/stage1/feature_selection.json"
echo "  results/stage2/feature_selection.json"
echo "then run gen_setting_jobs.py locally to generate Phase 2's per-setting SLURM jobs"
echo "with the real valid-settings count for each stage, then run submit_tx2017_phase2.sh."
