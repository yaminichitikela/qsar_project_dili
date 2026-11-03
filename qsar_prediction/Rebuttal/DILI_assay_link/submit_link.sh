#!/bin/bash
# Submits the DILI linking experiment -- the comparison this whole folder exists
# for: does knowing the in vitro HepG2 mechanisms improve DILI prediction?
#
#   link_prep                                   [builds the DILI side]
#     -> search per (arm, family, stage)
#       -> fit array per (arm, family, stage), 5 repeats
#         -> aggregate (CIs + paired test between arms)
#
# Two arms, identical except their inputs:
#   structure      ECFP4 / graphs only
#   structure_iv   the same, plus P(MMP active) and P(caspase active)
#
# Stage 1 = "Proxy-only" (clinical pool). Stage 2 = "Proxy+DILI" (adds NCV-888);
# graph families fine-tune from their stage-1 weights and inherit the stage-1
# config, so only the classical families search twice -- 26 searches, not 40.
#
# Usage:
#   ./submit_link.sh                    # everything
#   ./submit_link.sh --dry-run
#   ./submit_link.sh --from search      # DILI side already built
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs results

ARMS="structure structure_iv"
CLASSICAL="RF SVM XGB"
GRAPH="GCN GAT GIN SAGE Graphormer GraphGPS MVGNN"
N_REPEATS=5
DRY=0; FROM=prep
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --from) FROM="$2"; shift 2 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

sub() {
  local desc="$1"; shift
  if [ "$DRY" = "1" ]; then echo "[dry-run] $desc : sbatch $*" >&2; echo "000000"; return; fi
  local id; id=$(sbatch --parsable "$@")
  echo "  $desc -> $id" >&2
  echo "$id"
}

cd jobs
GATE=""
if [ "$FROM" = "prep" ]; then
  echo "== DILI-side prep"
  PREP=$(sub "link_prep" link_prep.slurm)
  GATE="--dependency=afterok:$PREP"
fi

ALL_FITS=""; n_search=0; n_tasks=0
for arm in $ARMS; do
  for fam in $CLASSICAL $GRAPH; do
    case " $CLASSICAL " in *" $fam "*) kind=cpu; graph=0 ;; *) kind=gpu; graph=1 ;; esac

    # stage 1: every family searches
    s1=$(sub "search $arm/$fam/s1 ($kind)" --job-name="s1_${arm}_${fam}" \
             --export=ALL,ARM="$arm",FAMILY="$fam",STAGE=1 \
             $GATE "link_search_${kind}.slurm")
    n_search=$((n_search + 1))
    f1=$(sub "fit    $arm/$fam/s1 (${N_REPEATS} repeats)" --job-name="f1_${arm}_${fam}" \
             --export=ALL,ARM="$arm",FAMILY="$fam",STAGE=1 \
             --array=0-$((N_REPEATS - 1)) \
             --dependency=afterok:"$s1" "link_fit_${kind}.slurm")
    n_tasks=$((n_tasks + N_REPEATS))

    # stage 2: classical re-searches; graph families inherit the stage-1 config
    # and warm-start from the stage-1 weights, so they depend on the s1 FITS
    if [ "$graph" = "0" ]; then
      s2=$(sub "search $arm/$fam/s2 ($kind)" --job-name="s2_${arm}_${fam}" \
               --export=ALL,ARM="$arm",FAMILY="$fam",STAGE=2 \
               $GATE "link_search_${kind}.slurm")
      n_search=$((n_search + 1))
      dep="afterok:$s2"
    else
      dep="afterok:$f1"
    fi
    f2=$(sub "fit    $arm/$fam/s2 (${N_REPEATS} repeats)" --job-name="f2_${arm}_${fam}" \
             --export=ALL,ARM="$arm",FAMILY="$fam",STAGE=2 \
             --array=0-$((N_REPEATS - 1)) \
             --dependency="$dep" "link_fit_${kind}.slurm")
    n_tasks=$((n_tasks + N_REPEATS))
    ALL_FITS="${ALL_FITS:+$ALL_FITS:}$f1:$f2"
  done
done

echo "== aggregate (CIs + paired test between arms)"
sub "aggregate" --dependency=afterok:"$ALL_FITS" link_aggregate.slurm > /dev/null

echo
echo "Submitted: $([ "$FROM" = prep ] && echo '1 prep + ')$n_search searches + 40 fit arrays ($n_tasks tasks) + 1 aggregate"
echo
echo "Watch:    squeue -u \$USER -o \"%.12i %.28j %.2t %.10M\""
echo "Headline: results/link/linking_summary.csv  (Structure-only vs Structure+in vitro)"
