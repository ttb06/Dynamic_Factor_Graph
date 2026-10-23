#!/usr/bin/env bash
# Graph-constrained Pareto selection on the nine frozen confirmation pools.
#SBATCH --job-name=llmtd-graph-pareto
#SBATCH --partition=prioritized
#SBATCH --array=0-8%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
set +x

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SELECTOR="$A/scripts/graph_pareto_anchored_resampling.py"
AUDITOR="$A/scripts/audit_graph_consistency.py"

i=$SLURM_ARRAY_TASK_ID
dataset_index=$((i / 3)); seed_index=$((i % 3))
case "$dataset_index" in
  0)
    DATASET=shoppers; BASE=proj; OLD_SELECTOR=gt4; SEEDS=(20261461 20261462 20261463); SOURCE_TASKS=(0 1 2)
    GRAPH="$A/outputs/shoppers_dynamic_confirm_s20261381_frozen_state_1067703_3/final_graph.json"
    EVAL_GRAPH="$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json"
    EDGE_IDS='r1_014,seed_003,r2_006,r1_015,r1_005,r1_021,r1_013'
    ;;
  1)
    DATASET=magic; BASE=proj; OLD_SELECTOR=gt4; SEEDS=(20261464 20261465 20261466); SOURCE_TASKS=(3 4 5)
    GRAPH="$A/outputs/extended_bscg_magic_1081009/graph.json"; EVAL_GRAPH="$GRAPH"
    EDGE_IDS='bscg_000,bscg_001,bscg_002,bscg_003,bscg_004'
    ;;
  2)
    DATASET=diabetes; BASE=hybrid; OLD_SELECTOR=int20_t005; SEEDS=(20261467 20261468 20261469); SOURCE_TASKS=(6 7 8)
    GRAPH="$A/outputs/bootstrap_conditional_graph_diabetes_1070443_2/graph.json"; EVAL_GRAPH="$GRAPH"
    EDGE_IDS='bscg_011'
    ;;
  *) echo "invalid dataset index" >&2; exit 64 ;;
esac
SEED=${SEEDS[$seed_index]}; SOURCE_TASK=${SOURCE_TASKS[$seed_index]}
SOURCE="$A/outputs/multidataset_particle_confirm_${DATASET}_${BASE}_${OLD_SELECTOR}_s${SEED}_1111943_${SOURCE_TASK}"
POOL="$SOURCE/pool/samples/dense_factor.csv"; BACKBONE="$SOURCE/pool/samples/unconditional.csv"
ANCHOR="$SOURCE/winner/selected_indices.csv"; G_INDEX="$SOURCE/g_indices.csv"
TAIL_INDEX="$SOURCE/tail20/selected_indices.csv"
for path in "$POOL" "$BACKBONE" "$ANCHOR" "$G_INDEX" "$TAIL_INDEX" "$GRAPH" "$EVAL_GRAPH"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done

OUTROOT="$A/outputs/graph_pareto_screen_${DATASET}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"
{
  printf 'stage=graph_pareto_anchor_screen\njob_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$i"
  printf 'dataset=%s\nbase=%s\nold_selector=%s\nseed=%s\nsource=%s\n' "$DATASET" "$BASE" "$OLD_SELECTOR" "$SEED" "$SOURCE"
  printf 'host=%s\nstarted_at=%s\ngit_commit=%s\n' "$(hostname)" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$SELECTOR" "$AUDITOR" "$GRAPH" "$ANCHOR" "$POOL"
} > "$PROVENANCE"

COMMON=(--repo "$REPO" --dataset "$DATASET" --pool "$POOL" --backbone-pool "$BACKBONE"
  --graph "$GRAPH" --anchor-index "$ANCHOR" --include-edge-ids "$EDGE_IDS" --include-roles actuate
  --count 4096 --bins 16 --pair-bins 4 --max-pairs 24 --target-pair-quota 8
  --pair-weight 1.5 --coverage-weight .30 --generic-rake-iterations 5 --generic-rake-damping .5
  --prior-index "$G_INDEX" --prior-weight 1 --prior-index "$TAIL_INDEX" --prior-weight 2
  --consensus-strength 4 --weight-clip 20 --temperature .50 --tail-mix .10
  --draws 256 --seed "$SEED")
INTEGRATED=()
if [[ "$DATASET" == diabetes ]]; then
  INTEGRATED=(--geometric-coverage-weight 20 --geometric-coverage-k 16
    --geometric-coverage-tolerance 0 --integrated-coverage-weight 20
    --integrated-coverage-tolerance .005 --integrated-coverage-shortlist 8
    --integrated-coverage-folds 2)
fi
run_variant() {
  local name=$1 gain=$2 quality=$3 iterations=$4 damping=$5
  "$PY" -u "$SELECTOR" "${COMMON[@]}" "${INTEGRATED[@]}" \
    --required-graph-gain "$gain" --quality-relative-tolerance "$quality" \
    --graph-rake-iterations "$iterations" --graph-rake-damping "$damping" \
    --output "$OUTROOT/$name"
}
run_variant pa_g0_q10   0  .10  0 .10
run_variant pa_g10_q15 .10 .15  1 .10
run_variant pa_g25_q20 .25 .20  1 .15
run_variant pa_g50_q30 .50 .30  2 .20

AUDIT_ARGS=(--repo "$REPO" --dataset "$DATASET" --pool "$POOL" --graph "$GRAPH"
  --include-edge-ids "$EDGE_IDS" --include-roles actuate --count 4096
  --sample-index "old=$ANCHOR")
for method in pa_g0_q10 pa_g10_q15 pa_g25_q20 pa_g50_q30; do
  AUDIT_ARGS+=(--sample-index "$method=$OUTROOT/$method/selected_indices.csv")
done
"$PY" -u "$AUDITOR" "${AUDIT_ARGS[@]}" --output "$OUTROOT/graph_audit"

"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset "$DATASET" --graph "$EVAL_GRAPH" --max-rows 4096 --seed "$SEED" \
  --sample "B=$SOURCE/tail20/b_reference.csv" --sample "G=$SOURCE/tail20/g_reference.csv" \
  --sample "R=$SOURCE/tail20/uniform_control.csv" --sample "old=$SOURCE/winner/selected.csv" \
  --sample "pa_g0_q10=$OUTROOT/pa_g0_q10/selected.csv" \
  --sample "pa_g10_q15=$OUTROOT/pa_g10_q15/selected.csv" \
  --sample "pa_g25_q20=$OUTROOT/pa_g25_q20/selected.csv" \
  --sample "pa_g50_q30=$OUTROOT/pa_g50_q30/selected.csv" --output "$OUTROOT/eval"
