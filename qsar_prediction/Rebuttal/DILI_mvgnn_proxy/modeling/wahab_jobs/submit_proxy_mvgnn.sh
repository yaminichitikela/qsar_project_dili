#!/bin/bash
# Submits prep_proxy.slurm (builds clin_train/clin_val/NCV-888/held-223 atom+
# bond graphs) -> mvgnn_stage1.slurm (train from scratch on clin_train,
# "Proxy-only") -> mvgnn_stage2.slurm (fine-tune Stage-1 weights on
# clin_train+NCV-888, "Proxy+DILI") -> consensus_proxy.slurm (combines with
# the existing RF/XGB/Graphormer/GraphGPS predictions into Consensus-MVGNN).
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

PREP_ID=$(sbatch --parsable prep_proxy.slurm)
echo "Submitted prep_proxy.slurm as job $PREP_ID"

S1_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID mvgnn_stage1.slurm)
echo "Submitted mvgnn_stage1.slurm as job $S1_ID (waits for $PREP_ID)"

S2_ID=$(sbatch --parsable --dependency=afterok:$S1_ID mvgnn_stage2.slurm)
echo "Submitted mvgnn_stage2.slurm as job $S2_ID (waits for $S1_ID)"

CONS_ID=$(sbatch --parsable --dependency=afterok:$S2_ID consensus_proxy.slurm)
echo "Submitted consensus_proxy.slurm as job $CONS_ID (waits for $S2_ID)"

echo
echo "All DILI_mvgnn_proxy jobs submitted (4 jobs, strictly sequential: prep -> stage1 -> stage2 -> consensus)."
echo "Check status with: squeue -u \$USER"
