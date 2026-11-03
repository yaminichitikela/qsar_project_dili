#!/bin/bash
# Runs the whole 1:5 scaffold pipeline on a local machine (CPU), in parallel.
# Used instead of submit_all.sh because the 1:5 training sets are small (~700).
cd "$(dirname "$0")"
mkdir -p logs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
pids=()
for m in rf svm xgb; do python3 train_classical.py scaffold $m > logs/local_$m.log 2>&1 & pids+=($!); done
for a in GCN GAT GIN SAGE; do python3 train_gnn.py scaffold $a > logs/local_$(echo $a | tr A-Z a-z).log 2>&1 & pids+=($!); done
python3 train_mvgnn.py scaffold > logs/local_mvgnn.log 2>&1 & pids+=($!)
python3 train_gt.py scaffold Graphormer > logs/local_graphormer.log 2>&1 & pids+=($!)
python3 train_gt.py scaffold GraphGPS > logs/local_graphgps.log 2>&1 & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] && python3 compute_consensus.py scaffold > logs/local_consensus.log 2>&1
echo "done fail=$fail"
