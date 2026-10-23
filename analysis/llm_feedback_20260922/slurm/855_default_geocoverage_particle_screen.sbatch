#!/usr/bin/env bash
# Train-only geometric-coverage constrained selection on six fresh gt4 pools.
#SBATCH --job-name=llmtd-geocov-screen
#SBATCH --partition=prioritized
#SBATCH --array=0-23%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
RESAMPLER="$A/scripts/balanced_particle_resampling.py"
GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
VARIANTS=(geo1_t0 geo5_t0 geo20_t0 geo5_t005)
SEEDS=(20261411 20261412 20261413 20261414 20261415 20261416)
i=$SLURM_ARRAY_TASK_ID
seed_index=$((i / 4))
variant_index=$((i % 4))
SEED=${SEEDS[$seed_index]}
VARIANT=${VARIANTS[$variant_index]}
SOURCE="$A/outputs/default_gt4_confirm_s${SEED}_1111735_${seed_index}"
GEOMETRY_WEIGHT=5
GEOMETRY_TOLERANCE=0
case "$VARIANT" in
  geo1_t0) GEOMETRY_WEIGHT=1 ;;
  geo5_t0) ;;
  geo20_t0) GEOMETRY_WEIGHT=20 ;;
  geo5_t005) GEOMETRY_TOLERANCE=.005 ;;
  *) echo "unknown variant" >&2; exit 64 ;;
esac

OUTROOT="$A/outputs/default_geocov_${VARIANT}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
for path in "$SOURCE/pool/samples/dense_factor.csv" "$SOURCE/g_indices.csv" "$SOURCE/tail20/selected_indices.csv"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done
{
  printf 'stage=Default_geometric_coverage_particle_screen\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=default\nvariant=%s\nmode=consensus_rake\nseed=%s\n' "$VARIANT" "$SEED"
  printf 'pool_samples=16384\nselected_samples=8192\nsteps=50\ndraws=128\n'
  printf 'geometric_coverage_weight=%s\ngeometric_coverage_k=16\ngeometric_coverage_tolerance=%s\n' "$GEOMETRY_WEIGHT" "$GEOMETRY_TOLERANCE"
  printf 'source=%s\nhost=%s\nstarted_at=%s\n' "$SOURCE" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$RESAMPLER" "$SOURCE/pool/samples/dense_factor.csv" "$SOURCE/g_indices.csv" "$SOURCE/tail20/selected_indices.csv"
} > "$PROVENANCE"

START=$(date +%s)
"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset default \
  --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --backbone-pool "$SOURCE/pool/samples/unconditional.csv" \
  --count 8192 --mode consensus_rake --bins 16 --pair-bins 4 \
  --max-pairs 24 --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .50 --tail-mix .10 --draws 128 --consensus-strength 4 \
  --prior-index "$SOURCE/g_indices.csv" --prior-weight 1 \
  --prior-index "$SOURCE/tail20/selected_indices.csv" --prior-weight 2 \
  --geometric-coverage-weight "$GEOMETRY_WEIGHT" --geometric-coverage-k 16 \
  --geometric-coverage-tolerance "$GEOMETRY_TOLERANCE" \
  --seed "$SEED" --output "$OUTROOT/resampled"
AFTER_RESAMPLE=$(date +%s)
printf 'pre_eval_runtime_seconds=%s\n' "$((AFTER_RESAMPLE - START))" >> "$PROVENANCE"

"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$GRAPH" \
  --sample "B=$SOURCE/control/b_reference.csv" \
  --sample "G=$SOURCE/control/g_reference.csv" \
  --sample "R=$SOURCE/control/selected.csv" \
  --sample "$VARIANT=$OUTROOT/resampled/selected.csv" \
  --seed "$SEED" --output "$OUTROOT/eval"
