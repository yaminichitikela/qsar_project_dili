#!/bin/bash
# Phase 1 of the CODE-CUTOFFS variant of the TX2017-Structure pipeline:
# feature_selection.py 1 (Proxy-only) and feature_selection.py 2
# (Proxy+DILI), using Xu et al.'s actual GitHub-code cutoff ranges (Fisher
# 0.01-0.05, AUC 0.51-0.60, RF/XGB top-100-to-1000) instead of the paper-text
# ranges used in the sibling DILI_tx2017 folder.
#
# No prep step here -- the underlying InChIKey-cleaned ECFP4 features are
# IDENTICAL to the paper-text run (only the feature-selection cutoffs
# differ), so this folder reuses ~/dili_tx2017_clean/data/ directly instead
# of paying for another 2-hour rdkit prep job. Copy it over once before
# running this script:
#   ssh <host> "bash -c 'cp -r ~/dili_tx2017_clean/data ~/dili_tx2017_code/data'"
#
# The number of VALID (non-empty) feature-selection settings per stage is
# only known after this phase finishes (some AUC-ROC cutoffs may select 0
# features). Do not generate/submit the per-setting jobs until you've pulled
# results/stage1/feature_selection.json and results/stage2/feature_selection.json
# back from Wahab and confirmed the real counts -- see gen_setting_jobs.py.
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

if [ ! -d "../data" ]; then
    echo "ERROR: ../data does not exist. Copy it from ~/dili_tx2017_clean/data/ first (see comment above)." >&2
    exit 1
fi

FS1_ID=$(sbatch --parsable feature_selection_stage1.slurm)
echo "Submitted feature_selection_stage1.slurm as job $FS1_ID"

FS2_ID=$(sbatch --parsable feature_selection_stage2.slurm)
echo "Submitted feature_selection_stage2.slurm as job $FS2_ID"

echo
echo "Phase 1 submitted (feature_selection x2 = 2 jobs, no prep needed). Check status with: squeue -u \$USER"
echo "Once both feature_selection jobs finish, pull back:"
echo "  results/stage1/feature_selection.json"
echo "  results/stage2/feature_selection.json"
echo "then run gen_setting_jobs.py locally to generate Phase 2's per-setting SLURM jobs"
echo "with the real valid-settings count for each stage, then run submit_tx2017_phase2.sh."
