#!/bin/bash
# Submits the corrected graph-transformer retrain, chained with --dependency=afterok:
#
#   check_inputs                     [gate: a failure stops everything below]
#     -> Graphormer array per (assay, protocol)
#        GraphGPS   array per (assay, protocol)
#          -> consensus per (assay, protocol)      (needs both architectures)
#            -> aggregate (once, across everything)
#
# One array per (assay, architecture, protocol), which is the per-architecture job
# convention used elsewhere in this project. Classical and GNN families are NOT
# retrained: the consensus reuses their published predictions.
#
# Usage:
#   ./submit_corrected.sh                  # all three assays, 82 units x 2 archs
#   ./submit_corrected.sh mmp              # one assay (recommended first)
#   ./submit_corrected.sh mmp loo          # one assay + protocol
#   ./submit_corrected.sh --dry-run mmp
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs results

ALL_ASSAYS="mmp hepg2 chok1"
ARCHS="Graphormer GraphGPS"
ALL_PROTOS="random_repeat loo scaffold up_single up_loo"

# The scope lives in data_map.py and is read from there rather than duplicated
# here: a second copy of the assay/protocol/unit table is exactly the kind of
# drift that made the first Wahab check report 0/82.
PY=${PYTHON:-python3}
# PYTHONPATH is pinned to ROOT because these are called after cd jobs
protos_for() { PYTHONPATH="$ROOT" $PY -c "import data_map as d; print(' '.join(d.ASSAY_PROTOCOLS['$1']))"; }
units_for()  { PYTHONPATH="$ROOT" $PY -c "import data_map as d; print(d.PROTOCOLS['$1']['n_units'])"; }

DRY=0; ARGS=()
for a in "$@"; do
  if [ "$a" = "--dry-run" ]; then DRY=1; else ARGS+=("$a"); fi
done

ASSAYS="$ALL_ASSAYS"; ONLY_PROTO=""
for a in "${ARGS[@]}"; do
  case " $ALL_ASSAYS " in *" $a "*) ASSAYS="$a"; continue ;; esac
  case " $ALL_PROTOS " in *" $a "*) ONLY_PROTO="$a" ;; esac
done

sub() {
  local desc="$1"; shift
  if [ "$DRY" = "1" ]; then echo "[dry-run] $desc : sbatch $*" >&2; echo "000000"; return; fi
  local id; id=$(sbatch --parsable "$@")
  echo "  $desc -> $id" >&2
  echo "$id"
}

cd jobs
echo "== input check (gates all GPU work)"
CHECK=$(sub "check" --export=ALL,ASSAYS="$(echo $ASSAYS)" check.slurm)

AGG_DEPS=""; n_arrays=0; n_tasks=0
for assay in $ASSAYS; do
  for proto in $(protos_for "$assay"); do
    if [ -n "$ONLY_PROTO" ] && [ "$proto" != "$ONLY_PROTO" ]; then continue; fi
    n=$(units_for "$proto")
    TRAIN_IDS=""
    for arch in $ARCHS; do
      id=$(sub "train $assay/$arch/$proto (${n} units)" \
               --job-name="g_${assay}_${arch}_${proto}" \
               --export=ALL,ASSAY="$assay",ARCH="$arch",PROTOCOL="$proto" \
               --array=0-$((n - 1)) \
               --dependency=afterok:"$CHECK" train_gt.slurm)
      TRAIN_IDS="${TRAIN_IDS:+$TRAIN_IDS:}$id"
      n_arrays=$((n_arrays + 1)); n_tasks=$((n_tasks + n))
    done
    # the consensus needs BOTH architectures for this protocol
    cid=$(sub "consensus $assay/$proto" --job-name="c_${assay}_${proto}" \
              --export=ALL,ASSAY="$assay",PROTOCOL="$proto" \
              --dependency=afterok:"$TRAIN_IDS" consensus.slurm)
    AGG_DEPS="${AGG_DEPS:+$AGG_DEPS:}$cid"
  done
done

echo "== aggregate"
sub "aggregate" --dependency=afterok:"$AGG_DEPS" aggregate.slurm > /dev/null

echo
echo "Submitted: 1 check + $n_arrays training arrays ($n_tasks tasks) + consensus + aggregate"
echo
echo "Watch:     squeue -u \$USER"
echo "Failures:  sacct -u \$USER --format=JobID,JobName%24,State,Elapsed | grep -E 'FAILED|TIMEOUT'"
echo "Headline:  results/old_vs_new_all.csv   (what correcting the architecture did)"
