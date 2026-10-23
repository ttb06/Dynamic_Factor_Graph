#!/usr/bin/env bash
# Reuse confirmation pools for multi-seed particle-selection ablations.
#SBATCH --job-name=llmtd-particle-ablation
#SBATCH --partition=prioritized
#SBATCH --array=0-5%6
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
SEEDS=(20261381 20261382 20261383 20261384 20261385 20261386)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
SOURCE="$A/outputs/default_particle_confirm_ratio_xgb50_s${SEED}_1110956_${SLURM_ARRAY_TASK_ID}"
OUTROOT="$A/outputs/default_particle_ablation_s${SEED}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT

if [[ ! -f "$SOURCE/resampled/selected.csv" ]]; then
  echo "missing completed source: $SOURCE" >&2
  exit 66
fi

{
  printf 'stage=Default_balanced_particle_multiseed_ablation\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=default\nseed=%s\npool_samples=16384\nselected_samples=8192\nsteps=50\n' "$SEED"
  printf 'source=%s\nhost=%s\nstarted_at=%s\n' "$SOURCE" "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$RESAMPLER" "$SOURCE/pool/samples/dense_factor.csv" "$SOURCE/pool/samples/unconditional.csv" "$SOURCE/resampled/selected.csv"
} > "$PROVENANCE"

"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset default \
  --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --backbone-pool "$SOURCE/pool/samples/unconditional.csv" \
  --count 8192 --mode uniform_search --bins 16 --pair-bins 4 --max-pairs 16 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .50 --tail-mix .10 --draws 32 --seed "$SEED" \
  --output "$OUTROOT/uniform32"

"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset default \
  --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --backbone-pool "$SOURCE/pool/samples/unconditional.csv" \
  --count 8192 --mode rake_marginal --bins 16 --pair-bins 4 --max-pairs 0 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .35 --tail-mix .10 --draws 32 --seed "$SEED" \
  --output "$OUTROOT/rake_marg35"

"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$GRAPH" \
  --sample "B=$SOURCE/resampled/b_reference.csv" \
  --sample "G=$SOURCE/resampled/g_reference.csv" \
  --sample "R=$SOURCE/resampled/uniform_control.csv" \
  --sample "ratio_xgb50=$SOURCE/resampled/selected.csv" \
  --sample "uniform32=$OUTROOT/uniform32/selected.csv" \
  --sample "rake_marg35=$OUTROOT/rake_marg35/selected.csv" \
  --seed "$SEED" --output "$OUTROOT/eval"
