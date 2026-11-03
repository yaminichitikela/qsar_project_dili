#!/bin/bash
# Submits the full InChIKey-cleaned DILI Proxy-only/Proxy+DILI pipeline:
#   prep -> {classical, GCN, GAT, GIN, SAGE, Graphormer, GraphGPS, MV-GNN} Stage-1
#        -> matching Stage-2 (fine-tune, except classical which re-searches
#           independently and only depends on prep)
#        -> consensus (waits for ALL 16 stage1+stage2 jobs explicitly, not just
#           the stage2 leaves -- classical's two stages run in parallel and
#           neither depends on the other, so both must be listed explicitly)
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

PREP_ID=$(sbatch --parsable prep.slurm)
echo "Submitted prep.slurm as job $PREP_ID"

CLASSICAL_S1_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID classical_stage1.slurm)
echo "Submitted classical_stage1.slurm as job $CLASSICAL_S1_ID"
CLASSICAL_S2_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID classical_stage2.slurm)
echo "Submitted classical_stage2.slurm as job $CLASSICAL_S2_ID"

ALL_DEP_IDS=($CLASSICAL_S1_ID $CLASSICAL_S2_ID)

for arch in GCN GAT GIN SAGE; do
    larch=$(echo "$arch" | tr 'A-Z' 'a-z')
    S1_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID "gnn_stage1_${arch}.slurm")
    echo "Submitted gnn_stage1_${arch}.slurm as job $S1_ID"
    S2_ID=$(sbatch --parsable --dependency=afterok:$S1_ID "gnn_stage2_${arch}.slurm")
    echo "Submitted gnn_stage2_${arch}.slurm as job $S2_ID (waits for $S1_ID)"
    ALL_DEP_IDS+=($S1_ID $S2_ID)
done

for arch in Graphormer GraphGPS; do
    S1_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID "gt_stage1_${arch}.slurm")
    echo "Submitted gt_stage1_${arch}.slurm as job $S1_ID"
    S2_ID=$(sbatch --parsable --dependency=afterok:$S1_ID "gt_stage2_${arch}.slurm")
    echo "Submitted gt_stage2_${arch}.slurm as job $S2_ID (waits for $S1_ID)"
    ALL_DEP_IDS+=($S1_ID $S2_ID)
done

MVGNN_S1_ID=$(sbatch --parsable --dependency=afterok:$PREP_ID mvgnn_stage1.slurm)
echo "Submitted mvgnn_stage1.slurm as job $MVGNN_S1_ID"
MVGNN_S2_ID=$(sbatch --parsable --dependency=afterok:$MVGNN_S1_ID mvgnn_stage2.slurm)
echo "Submitted mvgnn_stage2.slurm as job $MVGNN_S2_ID (waits for $MVGNN_S1_ID)"
ALL_DEP_IDS+=($MVGNN_S1_ID $MVGNN_S2_ID)

DEP=$(IFS=:; echo "${ALL_DEP_IDS[*]}")
CONS_ID=$(sbatch --parsable --dependency=afterok:$DEP consensus.slurm)
echo "Submitted consensus.slurm as job $CONS_ID (waits for all 16 model jobs: $DEP)"

echo
echo "All DILI_proxy_clean jobs submitted (1 prep + 16 model jobs + 1 consensus = 18 jobs)."
echo "Check status with: squeue -u \$USER"
