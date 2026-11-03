#!/bin/bash
# Submits the whole assay side of the linking experiment, chained with
# --dependency=afterok so nothing runs on bad inputs:
#
#   prep (mmp, caspase, dili)
#     -> leakage audit            [gate: a failure stops everything below]
#       -> per-family Optuna search        (2 assays x 10 families = 20 jobs)
#         -> per-family fit array          (2 x 10 x 21 repeats = 420 tasks)
#           -> out-of-bag feature build + re-audit
#
# One submission per (assay, family) stage, matching the per-architecture job
# convention in DILI_proxy_clean/modeling/wahab_jobs/jobs/.
#
# Usage:
#   ./submit_assays.sh                 # everything
#   ./submit_assays.sh --from search   # skip prep+audit (data already built)
#   ./submit_assays.sh --only mmp      # one assay
#   ./submit_assays.sh --dry-run       # print what would be submitted
#   ./submit_assays.sh --from search --families "SVM GCN GAT GIN SAGE Graphormer GraphGPS MVGNN"
#                                      # only the families still missing
#   ./submit_assays.sh --from search --missing
#                                      # same, worked out from results/<assay>/probs/
set -e
cd "$(dirname "$0")"
mkdir -p logs results

CLASSICAL="RF SVM XGB"
GRAPH="GCN GAT GIN SAGE Graphormer GraphGPS MVGNN"
ASSAYS="mmp caspase"
N_REPEATS=21          # must match prepare_assay_oob.py (R_BASE + patches)
FROM="prep"; DRY=0; FAMILIES=""; MISSING=0

while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ASSAYS="$2"; shift 2 ;;
    --families) FAMILIES="$2"; shift 2 ;;
    --missing) MISSING=1; shift ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

# --missing: a family counts as done only if it has all N_REPEATS fit outputs
# for every requested assay; anything short of that is resubmitted in full.
if [ "$MISSING" = "1" ]; then
  FAMILIES=""
  for fam in $CLASSICAL $GRAPH; do
    done_everywhere=1
    for assay in $ASSAYS; do
      n=$(ls "results/$assay/probs/${fam}_r"*_dili.npy 2>/dev/null | wc -l | tr -d ' ')
      [ "$n" -lt "$N_REPEATS" ] && done_everywhere=0
    done
    [ "$done_everywhere" = "0" ] && FAMILIES="${FAMILIES:+$FAMILIES }$fam"
  done
  echo "== --missing resolved to: ${FAMILIES:-<none, all complete>}"
  [ -z "$FAMILIES" ] && exit 0
fi

# restrict the two family lists to what was asked for, keeping cpu/gpu routing
if [ -n "$FAMILIES" ]; then
  keep() { local out=""; for f in $1; do case " $FAMILIES " in *" $f "*) out="${out:+$out }$f";; esac; done; echo "$out"; }
  CLASSICAL=$(keep "$CLASSICAL"); GRAPH=$(keep "$GRAPH")
fi

sub() {  # sub <description> <sbatch args...>
  local desc="$1"; shift
  if [ "$DRY" = "1" ]; then echo "[dry-run] $desc : sbatch $*" >&2; echo "000000"; return; fi
  local id; id=$(sbatch --parsable "$@")
  echo "  submitted $desc as $id" >&2
  echo "$id"
}

cd jobs
DEPS=""

# ── prep + audit ─────────────────────────────────────────────────────────────
if [ "$FROM" = "prep" ]; then
  echo "== prep"
  PREP_IDS=""
  for t in mmp caspase dili; do
    id=$(sub "prep/$t" --job-name="prep_$t" --export=ALL,TARGET="$t" prep.slurm)
    PREP_IDS="${PREP_IDS:+$PREP_IDS:}$id"
  done
  echo "== leakage audit (gates all training)"
  AUDIT_ID=$(sub "audit" --dependency=afterok:"$PREP_IDS" audit.slurm)
  GATE="--dependency=afterok:$AUDIT_ID"
else
  echo "== skipping prep/audit (--from $FROM); run audit_leakage.py yourself first"
  GATE=""
fi

# ── search + fit, one submission per (assay, family) ─────────────────────────
ALL_FIT_IDS=""
for assay in $ASSAYS; do
  for fam in $CLASSICAL $GRAPH; do
    case " $CLASSICAL " in *" $fam "*) kind=cpu ;; *) kind=gpu ;; esac
    tag="${assay}_${fam}"

    s_id=$(sub "search $tag ($kind)" --job-name="s_$tag" \
               --export=ALL,ASSAY="$assay",FAMILY="$fam" \
               $GATE "assay_search_${kind}.slurm")

    f_id=$(sub "fit    $tag ($kind, ${N_REPEATS} repeats)" --job-name="f_$tag" \
               --export=ALL,ASSAY="$assay",FAMILY="$fam" \
               --array=0-$((N_REPEATS - 1)) \
               --dependency=afterok:"$s_id" "assay_fit_${kind}.slurm")

    ALL_FIT_IDS="${ALL_FIT_IDS:+$ALL_FIT_IDS:}$f_id"
  done
done

# ── out-of-bag feature build, after every fit array ──────────────────────────
echo "== in vitro feature build"
IV_ID=$(sub "invitro" --dependency=afterok:"$ALL_FIT_IDS" invitro.slurm)

echo
echo "Submitted: 3 prep + 1 audit + 20 search + 20 fit arrays (${N_REPEATS} tasks each) + 1 feature build"
echo "           = $((3 + 1 + 20 + 1)) jobs, $((20 * N_REPEATS)) array tasks"
echo
echo "Watch:    squeue -u \$USER"
echo "Failures: sacct -u \$USER --format=JobID,JobName%20,State,Elapsed | grep -v COMPLETED"
echo "Result:   results/invitro_features.csv  (then rsync it back)"
