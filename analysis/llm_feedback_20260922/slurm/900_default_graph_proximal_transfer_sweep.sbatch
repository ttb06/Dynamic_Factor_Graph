#!/usr/bin/env bash
# Default transfer on three locked fresh pools. No new diffusion sampling.
# Keeps the original two-fold coverage barrier active during graph repair.
#SBATCH --job-name=llmtd-default-gp-xfer
#SBATCH --partition=prioritized
#SBATCH --array=0-2%3
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
SELECTOR="$A/scripts/graph_proximal_swap_repair.py"
AUDITOR="$A/scripts/audit_graph_consistency.py"
GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
EDGE_IDS='old::r1_001,old::r1_014,old::r1_015,old::r1_017'
SEEDS=(20261421 20261422 20261423)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SOURCE="$A/outputs/default_intcov_confirm_s${SEED}_1111817_${SLURM_ARRAY_TASK_ID}"
POOL="$SOURCE/pool/samples/dense_factor.csv"
BACKBONE="$SOURCE/pool/samples/unconditional.csv"
ANCHOR="$SOURCE/int20_t005/selected_indices.csv"
for path in "$POOL" "$BACKBONE" "$ANCHOR" "$GRAPH"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done

OUTROOT="$A/outputs/graph_proximal_transfer_default_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"
{
  printf 'stage=default_graph_proximal_transfer_screen\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=default\nbase=hybrid\nold_selector=int20_t005\nseed=%s\nsource=%s\n' "$SEED" "$SOURCE"
  printf 'pool_samples=16384\nselected_samples=8192\nedge_ids=%s\n' "$EDGE_IDS"
  printf 'coverage_barrier=geometric20+integrated20_t005_twofold\n'
  printf 'host=%s\nstarted_at=%s\ngit_commit=%s\n' "$(hostname)" "$(date --iso-8601=seconds)" "$(git rev-parse HEAD)"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$SELECTOR" "$AUDITOR" "$GRAPH" "$ANCHOR" "$POOL"
} > "$PROVENANCE"

COMMON=(--repo "$REPO" --dataset default --pool "$POOL" --backbone-pool "$BACKBONE"
  --graph "$GRAPH" --anchor-index "$ANCHOR" --include-edge-ids "$EDGE_IDS"
  --include-roles actuate --count 8192 --bins 16 --pair-bins 4 --max-pairs 24
  --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30
  --selection-priority quality --swap-temperature .75 --uniform-mix .10
  --coverage-boost 2 --hard-boost 2 --geometric-coverage-weight 20
  --geometric-coverage-k 16 --geometric-coverage-tolerance 0
  --integrated-coverage-weight 20 --integrated-coverage-tolerance .005
  --integrated-coverage-shortlist 8 --integrated-coverage-folds 2
  --draws 256 --seed "$SEED")
run_variant() {
  local name=$1 swap=$2 quality=$3 gain=$4
  "$PY" -u "$SELECTOR" "${COMMON[@]}" \
    --swap-fraction "$swap" --quality-relative-tolerance "$quality" \
    --required-graph-gain "$gain" --output "$OUTROOT/$name"
}
run_variant pqs0025_g05 .0025 .015 .05
run_variant pqs005_g10 .005 .03 .10
run_variant pqs010_g10 .010 .06 .10
run_variant pqs020_g25 .020 .12 .25

AUDIT_ARGS=(--repo "$REPO" --dataset default --pool "$POOL" --graph "$GRAPH"
  --include-edge-ids "$EDGE_IDS" --include-roles actuate --count 8192
  --sample-index "old=$ANCHOR")
for method in pqs0025_g05 pqs005_g10 pqs010_g10 pqs020_g25; do
  AUDIT_ARGS+=(--sample-index "$method=$OUTROOT/$method/selected_indices.csv")
done
"$PY" -u "$AUDITOR" "${AUDIT_ARGS[@]}" --output "$OUTROOT/graph_audit"
"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$GRAPH" --max-rows 8192 --seed "$SEED" \
  --sample "B=$SOURCE/control/b_reference.csv" \
  --sample "G=$SOURCE/control/g_reference.csv" \
  --sample "R=$SOURCE/control/uniform_control.csv" \
  --sample "old=$SOURCE/int20_t005/selected.csv" \
  --sample "pqs0025_g05=$OUTROOT/pqs0025_g05/selected.csv" \
  --sample "pqs005_g10=$OUTROOT/pqs005_g10/selected.csv" \
  --sample "pqs010_g10=$OUTROOT/pqs010_g10/selected.csv" \
  --sample "pqs020_g25=$OUTROOT/pqs020_g25/selected.csv" --output "$OUTROOT/eval"
