#!/usr/bin/env bash
# Multi-seed selection-only screen on the six already-generated fresh pools.
#SBATCH --job-name=llmtd-pareto-screen
#SBATCH --partition=prioritized
#SBATCH --array=0-47%12
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

VARIANTS=(pw075 pw150 pw300 cov150 tail20 temp50 target16 uniform150)
SEEDS=(20261391 20261392 20261393 20261394 20261395 20261396)
i=$SLURM_ARRAY_TASK_ID
seed_index=$((i / 8))
variant_index=$((i % 8))
SEED=${SEEDS[$seed_index]}
VARIANT=${VARIANTS[$variant_index]}
SOURCE="$A/outputs/default_rake_confirm_s${SEED}_1110985_${seed_index}"

MODE=rake_marginal_pair_select
PAIR_WEIGHT=1.50
COVERAGE_WEIGHT=.10
TEMPERATURE=.35
TAIL=.10
MAX_PAIRS=24
TARGET_QUOTA=8
case "$VARIANT" in
  pw075) PAIR_WEIGHT=.75 ;;
  pw150) ;;
  pw300) PAIR_WEIGHT=3.00 ;;
  cov150) COVERAGE_WEIGHT=.30 ;;
  tail20) COVERAGE_WEIGHT=.30; TAIL=.20 ;;
  temp50) TEMPERATURE=.50 ;;
  target16) MAX_PAIRS=32; TARGET_QUOTA=16 ;;
  uniform150) MODE=uniform_search; TEMPERATURE=.50 ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 64 ;;
esac

OUTROOT="$A/outputs/default_pareto_${VARIANT}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT

if [[ ! -s "$SOURCE/pool/samples/dense_factor.csv" || ! -s "$SOURCE/resampled/g_reference.csv" ]]; then
  echo "missing completed source pool: $SOURCE" >&2
  exit 66
fi

{
  printf 'stage=Default_train_only_pareto_particle_screen\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=default\nvariant=%s\nmode=%s\nseed=%s\n' "$VARIANT" "$MODE" "$SEED"
  printf 'pool_samples=16384\nselected_samples=8192\nsteps=50\ndraws=128\n'
  printf 'pair_weight=%s\ncoverage_weight=%s\ntemperature=%s\ntail_mix=%s\nmax_pairs=%s\ntarget_pair_quota=%s\n' \
    "$PAIR_WEIGHT" "$COVERAGE_WEIGHT" "$TEMPERATURE" "$TAIL" "$MAX_PAIRS" "$TARGET_QUOTA"
  printf 'source=%s\nhost=%s\nstarted_at=%s\n' "$SOURCE" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$RESAMPLER" "$SOURCE/pool/samples/dense_factor.csv" "$SOURCE/pool/samples/unconditional.csv"
} > "$PROVENANCE"

START=$(date +%s)
"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset default \
  --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --backbone-pool "$SOURCE/pool/samples/unconditional.csv" \
  --count 8192 --mode "$MODE" --bins 16 --pair-bins 4 \
  --max-pairs "$MAX_PAIRS" --target-pair-quota "$TARGET_QUOTA" \
  --pair-weight "$PAIR_WEIGHT" --coverage-weight "$COVERAGE_WEIGHT" \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature "$TEMPERATURE" --tail-mix "$TAIL" --draws 128 \
  --seed "$SEED" --output "$OUTROOT/resampled"
AFTER_RESAMPLE=$(date +%s)
printf 'pre_eval_runtime_seconds=%s\n' "$((AFTER_RESAMPLE - START))" >> "$PROVENANCE"

"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$GRAPH" \
  --sample "B=$SOURCE/resampled/b_reference.csv" \
  --sample "G=$SOURCE/resampled/g_reference.csv" \
  --sample "R=$SOURCE/resampled/uniform_control.csv" \
  --sample "$VARIANT=$OUTROOT/resampled/selected.csv" \
  --seed "$SEED" --output "$OUTROOT/eval"
