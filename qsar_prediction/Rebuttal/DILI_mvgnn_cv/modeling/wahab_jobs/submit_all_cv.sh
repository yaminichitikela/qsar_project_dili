#!/bin/bash
# Submits the full 10-model DILI CV pipeline (GNN family x4, GT family x2,
# classical x3, consensus). Requires prep_dili_cv.slurm (rewritten to build
# ECFP4 + gt-graph representations too) to have already finished --
# submit_dili_cv.sh (MV-GNN) already runs it; run that first, or run
# prep_dili_cv.slurm standalone if MV-GNN was already done under the old prep.
set -e
cd "$(dirname "$0")/jobs"
mkdir -p ../logs

ALL_DEP_IDS=()

# --- GNN: GCN ---
GCN_CFG_IDS=()
JID=$(sbatch --parsable "gnn_gcn_cfg0.slurm"); echo "Submitted gnn_gcn_cfg0.slurm as job $JID"; GCN_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gnn_gcn_cfg1.slurm"); echo "Submitted gnn_gcn_cfg1.slurm as job $JID"; GCN_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gnn_gcn_cfg2.slurm"); echo "Submitted gnn_gcn_cfg2.slurm as job $JID"; GCN_CFG_IDS+=($JID)
DEP=$(IFS=:; echo "${GCN_CFG_IDS[*]}")
GCN_SEL_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_gcn_select.slurm"); echo "Submitted gnn_gcn_select.slurm as job $GCN_SEL_ID"
GCN_S1_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GCN_SEL_ID "gnn_gcn_s1fold0.slurm"); echo "Submitted gnn_gcn_s1fold0.slurm as job $JID"; GCN_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GCN_SEL_ID "gnn_gcn_s1fold1.slurm"); echo "Submitted gnn_gcn_s1fold1.slurm as job $JID"; GCN_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GCN_SEL_ID "gnn_gcn_s1fold2.slurm"); echo "Submitted gnn_gcn_s1fold2.slurm as job $JID"; GCN_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GCN_SEL_ID "gnn_gcn_s1fold3.slurm"); echo "Submitted gnn_gcn_s1fold3.slurm as job $JID"; GCN_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GCN_SEL_ID "gnn_gcn_s1fold4.slurm"); echo "Submitted gnn_gcn_s1fold4.slurm as job $JID"; GCN_S1_IDS+=($JID)
DEP=$(IFS=:; echo "${GCN_S1_IDS[*]}")
GCN_AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_gcn_agg1.slurm"); echo "Submitted gnn_gcn_agg1.slurm as job $GCN_AGG1_ID"
GCN_S2_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GCN_AGG1_ID "gnn_gcn_s2fold0.slurm"); echo "Submitted gnn_gcn_s2fold0.slurm as job $JID"; GCN_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GCN_AGG1_ID "gnn_gcn_s2fold1.slurm"); echo "Submitted gnn_gcn_s2fold1.slurm as job $JID"; GCN_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GCN_AGG1_ID "gnn_gcn_s2fold2.slurm"); echo "Submitted gnn_gcn_s2fold2.slurm as job $JID"; GCN_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GCN_AGG1_ID "gnn_gcn_s2fold3.slurm"); echo "Submitted gnn_gcn_s2fold3.slurm as job $JID"; GCN_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GCN_AGG1_ID "gnn_gcn_s2fold4.slurm"); echo "Submitted gnn_gcn_s2fold4.slurm as job $JID"; GCN_S2_IDS+=($JID)
DEP=$(IFS=:; echo "${GCN_S2_IDS[*]}")
GCN_AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_gcn_agg2.slurm"); echo "Submitted gnn_gcn_agg2.slurm as job $GCN_AGG2_ID"
ALL_DEP_IDS+=($GCN_AGG1_ID $GCN_AGG2_ID)

# --- GNN: GAT ---
GAT_CFG_IDS=()
JID=$(sbatch --parsable "gnn_gat_cfg0.slurm"); echo "Submitted gnn_gat_cfg0.slurm as job $JID"; GAT_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gnn_gat_cfg1.slurm"); echo "Submitted gnn_gat_cfg1.slurm as job $JID"; GAT_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gnn_gat_cfg2.slurm"); echo "Submitted gnn_gat_cfg2.slurm as job $JID"; GAT_CFG_IDS+=($JID)
DEP=$(IFS=:; echo "${GAT_CFG_IDS[*]}")
GAT_SEL_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_gat_select.slurm"); echo "Submitted gnn_gat_select.slurm as job $GAT_SEL_ID"
GAT_S1_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GAT_SEL_ID "gnn_gat_s1fold0.slurm"); echo "Submitted gnn_gat_s1fold0.slurm as job $JID"; GAT_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GAT_SEL_ID "gnn_gat_s1fold1.slurm"); echo "Submitted gnn_gat_s1fold1.slurm as job $JID"; GAT_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GAT_SEL_ID "gnn_gat_s1fold2.slurm"); echo "Submitted gnn_gat_s1fold2.slurm as job $JID"; GAT_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GAT_SEL_ID "gnn_gat_s1fold3.slurm"); echo "Submitted gnn_gat_s1fold3.slurm as job $JID"; GAT_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GAT_SEL_ID "gnn_gat_s1fold4.slurm"); echo "Submitted gnn_gat_s1fold4.slurm as job $JID"; GAT_S1_IDS+=($JID)
DEP=$(IFS=:; echo "${GAT_S1_IDS[*]}")
GAT_AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_gat_agg1.slurm"); echo "Submitted gnn_gat_agg1.slurm as job $GAT_AGG1_ID"
GAT_S2_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GAT_AGG1_ID "gnn_gat_s2fold0.slurm"); echo "Submitted gnn_gat_s2fold0.slurm as job $JID"; GAT_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GAT_AGG1_ID "gnn_gat_s2fold1.slurm"); echo "Submitted gnn_gat_s2fold1.slurm as job $JID"; GAT_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GAT_AGG1_ID "gnn_gat_s2fold2.slurm"); echo "Submitted gnn_gat_s2fold2.slurm as job $JID"; GAT_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GAT_AGG1_ID "gnn_gat_s2fold3.slurm"); echo "Submitted gnn_gat_s2fold3.slurm as job $JID"; GAT_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GAT_AGG1_ID "gnn_gat_s2fold4.slurm"); echo "Submitted gnn_gat_s2fold4.slurm as job $JID"; GAT_S2_IDS+=($JID)
DEP=$(IFS=:; echo "${GAT_S2_IDS[*]}")
GAT_AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_gat_agg2.slurm"); echo "Submitted gnn_gat_agg2.slurm as job $GAT_AGG2_ID"
ALL_DEP_IDS+=($GAT_AGG1_ID $GAT_AGG2_ID)

# --- GNN: GIN ---
GIN_CFG_IDS=()
JID=$(sbatch --parsable "gnn_gin_cfg0.slurm"); echo "Submitted gnn_gin_cfg0.slurm as job $JID"; GIN_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gnn_gin_cfg1.slurm"); echo "Submitted gnn_gin_cfg1.slurm as job $JID"; GIN_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gnn_gin_cfg2.slurm"); echo "Submitted gnn_gin_cfg2.slurm as job $JID"; GIN_CFG_IDS+=($JID)
DEP=$(IFS=:; echo "${GIN_CFG_IDS[*]}")
GIN_SEL_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_gin_select.slurm"); echo "Submitted gnn_gin_select.slurm as job $GIN_SEL_ID"
GIN_S1_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GIN_SEL_ID "gnn_gin_s1fold0.slurm"); echo "Submitted gnn_gin_s1fold0.slurm as job $JID"; GIN_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GIN_SEL_ID "gnn_gin_s1fold1.slurm"); echo "Submitted gnn_gin_s1fold1.slurm as job $JID"; GIN_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GIN_SEL_ID "gnn_gin_s1fold2.slurm"); echo "Submitted gnn_gin_s1fold2.slurm as job $JID"; GIN_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GIN_SEL_ID "gnn_gin_s1fold3.slurm"); echo "Submitted gnn_gin_s1fold3.slurm as job $JID"; GIN_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GIN_SEL_ID "gnn_gin_s1fold4.slurm"); echo "Submitted gnn_gin_s1fold4.slurm as job $JID"; GIN_S1_IDS+=($JID)
DEP=$(IFS=:; echo "${GIN_S1_IDS[*]}")
GIN_AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_gin_agg1.slurm"); echo "Submitted gnn_gin_agg1.slurm as job $GIN_AGG1_ID"
GIN_S2_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GIN_AGG1_ID "gnn_gin_s2fold0.slurm"); echo "Submitted gnn_gin_s2fold0.slurm as job $JID"; GIN_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GIN_AGG1_ID "gnn_gin_s2fold1.slurm"); echo "Submitted gnn_gin_s2fold1.slurm as job $JID"; GIN_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GIN_AGG1_ID "gnn_gin_s2fold2.slurm"); echo "Submitted gnn_gin_s2fold2.slurm as job $JID"; GIN_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GIN_AGG1_ID "gnn_gin_s2fold3.slurm"); echo "Submitted gnn_gin_s2fold3.slurm as job $JID"; GIN_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GIN_AGG1_ID "gnn_gin_s2fold4.slurm"); echo "Submitted gnn_gin_s2fold4.slurm as job $JID"; GIN_S2_IDS+=($JID)
DEP=$(IFS=:; echo "${GIN_S2_IDS[*]}")
GIN_AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_gin_agg2.slurm"); echo "Submitted gnn_gin_agg2.slurm as job $GIN_AGG2_ID"
ALL_DEP_IDS+=($GIN_AGG1_ID $GIN_AGG2_ID)

# --- GNN: SAGE ---
SAGE_CFG_IDS=()
JID=$(sbatch --parsable "gnn_sage_cfg0.slurm"); echo "Submitted gnn_sage_cfg0.slurm as job $JID"; SAGE_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gnn_sage_cfg1.slurm"); echo "Submitted gnn_sage_cfg1.slurm as job $JID"; SAGE_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gnn_sage_cfg2.slurm"); echo "Submitted gnn_sage_cfg2.slurm as job $JID"; SAGE_CFG_IDS+=($JID)
DEP=$(IFS=:; echo "${SAGE_CFG_IDS[*]}")
SAGE_SEL_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_sage_select.slurm"); echo "Submitted gnn_sage_select.slurm as job $SAGE_SEL_ID"
SAGE_S1_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$SAGE_SEL_ID "gnn_sage_s1fold0.slurm"); echo "Submitted gnn_sage_s1fold0.slurm as job $JID"; SAGE_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$SAGE_SEL_ID "gnn_sage_s1fold1.slurm"); echo "Submitted gnn_sage_s1fold1.slurm as job $JID"; SAGE_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$SAGE_SEL_ID "gnn_sage_s1fold2.slurm"); echo "Submitted gnn_sage_s1fold2.slurm as job $JID"; SAGE_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$SAGE_SEL_ID "gnn_sage_s1fold3.slurm"); echo "Submitted gnn_sage_s1fold3.slurm as job $JID"; SAGE_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$SAGE_SEL_ID "gnn_sage_s1fold4.slurm"); echo "Submitted gnn_sage_s1fold4.slurm as job $JID"; SAGE_S1_IDS+=($JID)
DEP=$(IFS=:; echo "${SAGE_S1_IDS[*]}")
SAGE_AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_sage_agg1.slurm"); echo "Submitted gnn_sage_agg1.slurm as job $SAGE_AGG1_ID"
SAGE_S2_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$SAGE_AGG1_ID "gnn_sage_s2fold0.slurm"); echo "Submitted gnn_sage_s2fold0.slurm as job $JID"; SAGE_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$SAGE_AGG1_ID "gnn_sage_s2fold1.slurm"); echo "Submitted gnn_sage_s2fold1.slurm as job $JID"; SAGE_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$SAGE_AGG1_ID "gnn_sage_s2fold2.slurm"); echo "Submitted gnn_sage_s2fold2.slurm as job $JID"; SAGE_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$SAGE_AGG1_ID "gnn_sage_s2fold3.slurm"); echo "Submitted gnn_sage_s2fold3.slurm as job $JID"; SAGE_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$SAGE_AGG1_ID "gnn_sage_s2fold4.slurm"); echo "Submitted gnn_sage_s2fold4.slurm as job $JID"; SAGE_S2_IDS+=($JID)
DEP=$(IFS=:; echo "${SAGE_S2_IDS[*]}")
SAGE_AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP "gnn_sage_agg2.slurm"); echo "Submitted gnn_sage_agg2.slurm as job $SAGE_AGG2_ID"
ALL_DEP_IDS+=($SAGE_AGG1_ID $SAGE_AGG2_ID)

# --- GT: Graphormer ---
GRAPHORMER_CFG_IDS=()
JID=$(sbatch --parsable "gt_graphormer_cfg0.slurm"); echo "Submitted gt_graphormer_cfg0.slurm as job $JID"; GRAPHORMER_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gt_graphormer_cfg1.slurm"); echo "Submitted gt_graphormer_cfg1.slurm as job $JID"; GRAPHORMER_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gt_graphormer_cfg2.slurm"); echo "Submitted gt_graphormer_cfg2.slurm as job $JID"; GRAPHORMER_CFG_IDS+=($JID)
DEP=$(IFS=:; echo "${GRAPHORMER_CFG_IDS[*]}")
GRAPHORMER_SEL_ID=$(sbatch --parsable --dependency=afterok:$DEP "gt_graphormer_select.slurm"); echo "Submitted gt_graphormer_select.slurm as job $GRAPHORMER_SEL_ID"
GRAPHORMER_S1_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_SEL_ID "gt_graphormer_s1fold0.slurm"); echo "Submitted gt_graphormer_s1fold0.slurm as job $JID"; GRAPHORMER_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_SEL_ID "gt_graphormer_s1fold1.slurm"); echo "Submitted gt_graphormer_s1fold1.slurm as job $JID"; GRAPHORMER_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_SEL_ID "gt_graphormer_s1fold2.slurm"); echo "Submitted gt_graphormer_s1fold2.slurm as job $JID"; GRAPHORMER_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_SEL_ID "gt_graphormer_s1fold3.slurm"); echo "Submitted gt_graphormer_s1fold3.slurm as job $JID"; GRAPHORMER_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_SEL_ID "gt_graphormer_s1fold4.slurm"); echo "Submitted gt_graphormer_s1fold4.slurm as job $JID"; GRAPHORMER_S1_IDS+=($JID)
DEP=$(IFS=:; echo "${GRAPHORMER_S1_IDS[*]}")
GRAPHORMER_AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP "gt_graphormer_agg1.slurm"); echo "Submitted gt_graphormer_agg1.slurm as job $GRAPHORMER_AGG1_ID"
GRAPHORMER_S2_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_AGG1_ID "gt_graphormer_s2fold0.slurm"); echo "Submitted gt_graphormer_s2fold0.slurm as job $JID"; GRAPHORMER_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_AGG1_ID "gt_graphormer_s2fold1.slurm"); echo "Submitted gt_graphormer_s2fold1.slurm as job $JID"; GRAPHORMER_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_AGG1_ID "gt_graphormer_s2fold2.slurm"); echo "Submitted gt_graphormer_s2fold2.slurm as job $JID"; GRAPHORMER_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_AGG1_ID "gt_graphormer_s2fold3.slurm"); echo "Submitted gt_graphormer_s2fold3.slurm as job $JID"; GRAPHORMER_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHORMER_AGG1_ID "gt_graphormer_s2fold4.slurm"); echo "Submitted gt_graphormer_s2fold4.slurm as job $JID"; GRAPHORMER_S2_IDS+=($JID)
DEP=$(IFS=:; echo "${GRAPHORMER_S2_IDS[*]}")
GRAPHORMER_AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP "gt_graphormer_agg2.slurm"); echo "Submitted gt_graphormer_agg2.slurm as job $GRAPHORMER_AGG2_ID"
ALL_DEP_IDS+=($GRAPHORMER_AGG1_ID $GRAPHORMER_AGG2_ID)

# --- GT: GraphGPS ---
GRAPHGPS_CFG_IDS=()
JID=$(sbatch --parsable "gt_graphgps_cfg0.slurm"); echo "Submitted gt_graphgps_cfg0.slurm as job $JID"; GRAPHGPS_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gt_graphgps_cfg1.slurm"); echo "Submitted gt_graphgps_cfg1.slurm as job $JID"; GRAPHGPS_CFG_IDS+=($JID)
JID=$(sbatch --parsable "gt_graphgps_cfg2.slurm"); echo "Submitted gt_graphgps_cfg2.slurm as job $JID"; GRAPHGPS_CFG_IDS+=($JID)
DEP=$(IFS=:; echo "${GRAPHGPS_CFG_IDS[*]}")
GRAPHGPS_SEL_ID=$(sbatch --parsable --dependency=afterok:$DEP "gt_graphgps_select.slurm"); echo "Submitted gt_graphgps_select.slurm as job $GRAPHGPS_SEL_ID"
GRAPHGPS_S1_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_SEL_ID "gt_graphgps_s1fold0.slurm"); echo "Submitted gt_graphgps_s1fold0.slurm as job $JID"; GRAPHGPS_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_SEL_ID "gt_graphgps_s1fold1.slurm"); echo "Submitted gt_graphgps_s1fold1.slurm as job $JID"; GRAPHGPS_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_SEL_ID "gt_graphgps_s1fold2.slurm"); echo "Submitted gt_graphgps_s1fold2.slurm as job $JID"; GRAPHGPS_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_SEL_ID "gt_graphgps_s1fold3.slurm"); echo "Submitted gt_graphgps_s1fold3.slurm as job $JID"; GRAPHGPS_S1_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_SEL_ID "gt_graphgps_s1fold4.slurm"); echo "Submitted gt_graphgps_s1fold4.slurm as job $JID"; GRAPHGPS_S1_IDS+=($JID)
DEP=$(IFS=:; echo "${GRAPHGPS_S1_IDS[*]}")
GRAPHGPS_AGG1_ID=$(sbatch --parsable --dependency=afterok:$DEP "gt_graphgps_agg1.slurm"); echo "Submitted gt_graphgps_agg1.slurm as job $GRAPHGPS_AGG1_ID"
GRAPHGPS_S2_IDS=()
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_AGG1_ID "gt_graphgps_s2fold0.slurm"); echo "Submitted gt_graphgps_s2fold0.slurm as job $JID"; GRAPHGPS_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_AGG1_ID "gt_graphgps_s2fold1.slurm"); echo "Submitted gt_graphgps_s2fold1.slurm as job $JID"; GRAPHGPS_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_AGG1_ID "gt_graphgps_s2fold2.slurm"); echo "Submitted gt_graphgps_s2fold2.slurm as job $JID"; GRAPHGPS_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_AGG1_ID "gt_graphgps_s2fold3.slurm"); echo "Submitted gt_graphgps_s2fold3.slurm as job $JID"; GRAPHGPS_S2_IDS+=($JID)
JID=$(sbatch --parsable --dependency=afterok:$GRAPHGPS_AGG1_ID "gt_graphgps_s2fold4.slurm"); echo "Submitted gt_graphgps_s2fold4.slurm as job $JID"; GRAPHGPS_S2_IDS+=($JID)
DEP=$(IFS=:; echo "${GRAPHGPS_S2_IDS[*]}")
GRAPHGPS_AGG2_ID=$(sbatch --parsable --dependency=afterok:$DEP "gt_graphgps_agg2.slurm"); echo "Submitted gt_graphgps_agg2.slurm as job $GRAPHGPS_AGG2_ID"
ALL_DEP_IDS+=($GRAPHGPS_AGG1_ID $GRAPHGPS_AGG2_ID)

# --- Classical: RF/SVM/XGB, both stages (independent, CPU-only) ---
RF_S1_ID=$(sbatch --parsable "classical_rf_s1.slurm"); echo "Submitted classical_rf_s1.slurm as job $RF_S1_ID"
ALL_DEP_IDS+=($RF_S1_ID)
RF_S2_ID=$(sbatch --parsable "classical_rf_s2.slurm"); echo "Submitted classical_rf_s2.slurm as job $RF_S2_ID"
ALL_DEP_IDS+=($RF_S2_ID)
SVM_S1_ID=$(sbatch --parsable "classical_svm_s1.slurm"); echo "Submitted classical_svm_s1.slurm as job $SVM_S1_ID"
ALL_DEP_IDS+=($SVM_S1_ID)
SVM_S2_ID=$(sbatch --parsable "classical_svm_s2.slurm"); echo "Submitted classical_svm_s2.slurm as job $SVM_S2_ID"
ALL_DEP_IDS+=($SVM_S2_ID)
XGB_S1_ID=$(sbatch --parsable "classical_xgb_s1.slurm"); echo "Submitted classical_xgb_s1.slurm as job $XGB_S1_ID"
ALL_DEP_IDS+=($XGB_S1_ID)
XGB_S2_ID=$(sbatch --parsable "classical_xgb_s2.slurm"); echo "Submitted classical_xgb_s2.slurm as job $XGB_S2_ID"
ALL_DEP_IDS+=($XGB_S2_ID)

# --- Consensus: waits for every model above (MV-GNN already has its own
#     completed results from the earlier submit_dili_cv.sh run) ---
DEP=$(IFS=:; echo "${ALL_DEP_IDS[*]}")
CONS_ID=$(sbatch --parsable --dependency=afterok:$DEP consensus_cv.slurm)
echo "Submitted consensus_cv.slurm as job $CONS_ID (waits for all 9 non-MVGNN models: $DEP)"

echo
echo "All CV jobs submitted."
echo "Check status with: squeue -u \$USER"
