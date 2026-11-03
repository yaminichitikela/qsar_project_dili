#!/bin/bash
# Runs every TX2017 split-protocol unit sequentially (about one minute each,
# roughly 40 minutes in total). Safe to re-run: finished units are skipped.
set -u
cd "$(dirname "$0")"
# TX_SAMPLING=up (default) writes results/; TX_SAMPLING=down writes results_down/.
export TX_SAMPLING="${TX_SAMPLING:-up}"
if [ "$TX_SAMPLING" = "down" ]; then RES=results_down; else RES=results; fi
echo "sampling=$TX_SAMPLING -> $RES/"
for spec in "random_repeat 5" "cv 10" "scaffold 5"; do
  set -- $spec; proto=$1; n=$2
  for st in 1 2; do
    for ((u=0; u<n; u++)); do
      have=$(ls $RES/$proto/*_s${st}_u${u}_metrics.csv 2>/dev/null | wc -l)
      if [ "$have" -ge 5 ]; then echo "skip $proto s$st u$u"; continue; fi
      echo "=== $proto stage $st unit $u"
      python3 train_tx_unit.py "$proto" "$st" "$u" || echo "FAILED $proto s$st u$u"
    done
  done
done
echo "ALL DONE"
