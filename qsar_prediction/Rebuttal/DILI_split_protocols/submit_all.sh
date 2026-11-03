#!/bin/bash
# Submits the DILI split-protocol experiment on Wahab.
#
#   per protocol x family:  Stage-1 array (one task per unit)
#                           -> Stage-2 array (graph families: each task waits for
#                              the SAME unit's Stage-1 task via aftercorr, because
#                              it warm-starts from those weights; classical
#                              families have no weights and start immediately)
#   -> one aggregate job after everything (afterany)
#
# Splits are pre-built (splits/), configs are fixed (configs/), no search runs.
#
# Usage:
#   ./submit_all.sh                                  # all 4 protocols
#   ./submit_all.sh --protocols "cv scaffold"        # a subset
#   ./submit_all.sh --families "Graphormer GraphGPS" # a subset of families
#   ./submit_all.sh --dry-run
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs results

PROTOCOLS="random_repeat random_repeat_nodown cv scaffold random_repeat_up cv_up scaffold_up"
FAMILIES="RF SVM XGB GCN GAT GIN SAGE Graphormer GraphGPS MVGNN"
DRY=0
FORCE=0        # by default, a protocol/family/stage whose units all finished is skipped
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --protocols) PROTOCOLS="$2"; shift 2 ;;
    --families) FAMILIES="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

units() {
  case "$1" in
    cv|cv_up) echo 10 ;;
    random_repeat|random_repeat_nodown|scaffold|random_repeat_up|scaffold_up) echo 5 ;;
    *) echo "unknown protocol $1" >&2; exit 1 ;;
  esac
}

# already finished? (all N per-unit metric files present for this stage)
done_already() {
  [ "$FORCE" = "1" ] && return 1
  local proto="$1" fam="$2" stage="$3" n="$4"
  local have
  have=$(ls "$ROOT/results/$proto/${fam}_s${stage}_u"*_metrics.csv 2>/dev/null | wc -l)
  [ "$have" -ge "$n" ]
}

sub() {
  local desc="$1"; shift
  if [ "$DRY" = "1" ]; then echo "[dry-run] $desc : sbatch $*" >&2; echo "000000"; return; fi
  local id; id=$(sbatch --parsable "$@")
  echo "  $desc -> $id" >&2
  echo "$id"
}

# fail early on the laptop-vs-Wahab data mismatch rather than in 500 jobs
python3 -c "import common; common.check_clin_order(common.data_dir()); print('data:', common.data_dir())"
[ -f splits/split_report.csv ] || { echo "splits/ missing -- run prepare_splits.py"; exit 1; }

cd jobs
ALL=""; n_tasks=0; n_skipped=0
for proto in $PROTOCOLS; do
  N=$(units "$proto")
  for fam in $FAMILIES; do
    case "$fam" in RF|SVM|XGB) kind=cpu; graph=0 ;; *) kind=gpu; graph=1 ;; esac
    tag="${proto}_${fam}"
    # skip work that already finished, so re-running this script only submits
    # what is missing (pass --force to resubmit everything)
    if done_already "$proto" "$fam" 1 "$N" && done_already "$proto" "$fam" 2 "$N"; then
      n_skipped=$((n_skipped + 2 * N))
      continue
    fi
    f1=$(sub "s1 $tag (${N} units, $kind)" --job-name="d1_$tag" \
             --export=ALL,PROTOCOL="$proto",FAMILY="$fam",STAGE=1 \
             --array=0-$((N - 1)) "fit_${kind}.slurm")
    if [ "$graph" = "1" ]; then dep="--dependency=aftercorr:$f1"; else dep=""; fi
    f2=$(sub "s2 $tag (${N} units, $kind)" --job-name="d2_$tag" \
             --export=ALL,PROTOCOL="$proto",FAMILY="$fam",STAGE=2 \
             --array=0-$((N - 1)) $dep "fit_${kind}.slurm")
    n_tasks=$((n_tasks + 2 * N))
    ALL="${ALL:+$ALL:}$f1:$f2"
  done
done

if [ -z "$ALL" ]; then
  echo "Nothing to submit: every requested protocol/family already has results."
  echo "Use --force to resubmit."
  exit 0
fi

sub "aggregate" --dependency=afterany:"$ALL" aggregate.slurm > /dev/null
echo
echo "Submitted $n_tasks fit tasks + 1 aggregate (skipped $n_skipped already-finished tasks)."
echo "Watch:   squeue -u \$USER -o \"%.12i %.30j %.2t %.10M\""
echo "Results: results/summary.csv, results/comparison_vs_baseline.csv,"
echo "         results/tables_dili_split_protocols.tex"
