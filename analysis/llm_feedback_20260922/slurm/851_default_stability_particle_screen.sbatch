#!/usr/bin/env bash
# Stability-selection coreset screen using prior selections on identical pools.
#SBATCH --job-name=llmtd-stability-screen
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
VARIANTS=(gt4 gtu2 gtu4 gttu3)
SEEDS=(20261391 20261392 20261393 20261394 20261395 20261396)
i=$SLURM_ARRAY_TASK_ID
seed_index=$((i / 4))
variant_index=$((i % 4))
SEED=${SEEDS[$seed_index]}
VARIANT=${VARIANTS[$variant_index]}
SOURCE="$A/outputs/default_rake_confirm_s${SEED}_1110985_${seed_index}"
TAIL_TASK=$((seed_index * 8 + 4))
TEMP_TASK=$((seed_index * 8 + 5))
UNIFORM_TASK=$((seed_index * 8 + 7))
TAIL_INDEX="$A/outputs/default_pareto_tail20_s${SEED}_1111655_${TAIL_TASK}/resampled/selected_indices.csv"
TEMP_INDEX="$A/outputs/default_pareto_temp50_s${SEED}_1111655_${TEMP_TASK}/resampled/selected_indices.csv"
UNIFORM_INDEX="$A/outputs/default_pareto_uniform150_s${SEED}_1111655_${UNIFORM_TASK}/resampled/selected_indices.csv"
OUTROOT="$A/outputs/default_stability_${VARIANT}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
G_INDEX="$OUTROOT/g_indices.csv"
awk 'BEGIN { print "pool_index"; for (i=0; i<8192; i++) print i }' > "$G_INDEX"

STRENGTH=4
PRIOR_ARGS=(--prior-index "$G_INDEX" --prior-weight 1 --prior-index "$TAIL_INDEX" --prior-weight 2)
case "$VARIANT" in
  gt4) ;;
  gtu2) STRENGTH=2; PRIOR_ARGS+=(--prior-index "$UNIFORM_INDEX" --prior-weight 1) ;;
  gtu4) PRIOR_ARGS+=(--prior-index "$UNIFORM_INDEX" --prior-weight 1) ;;
  gttu3) STRENGTH=3; PRIOR_ARGS+=(--prior-index "$UNIFORM_INDEX" --prior-weight 1 --prior-index "$TEMP_INDEX" --prior-weight .5) ;;
  *) echo "unknown variant" >&2; exit 64 ;;
esac

PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
for path in "$SOURCE/pool/samples/dense_factor.csv" "$TAIL_INDEX"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 66; }
done
{
  printf 'stage=Default_stability_particle_screen\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=default\nvariant=%s\nmode=consensus_rake\nseed=%s\n' "$VARIANT" "$SEED"
  printf 'pool_samples=16384\nselected_samples=8192\nsteps=50\ndraws=128\nconsensus_strength=%s\n' "$STRENGTH"
  printf 'host=%s\nstarted_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)"
  sha256sum "$RESAMPLER" "$SOURCE/pool/samples/dense_factor.csv" "$G_INDEX" "$TAIL_INDEX"
} > "$PROVENANCE"

START=$(date +%s)
"$PY" -u "$RESAMPLER" --repo "$REPO" --dataset default \
  --pool "$SOURCE/pool/samples/dense_factor.csv" \
  --backbone-pool "$SOURCE/pool/samples/unconditional.csv" \
  --count 8192 --mode consensus_rake --bins 16 --pair-bins 4 \
  --max-pairs 24 --target-pair-quota 8 --pair-weight 1.5 --coverage-weight .30 \
  --rake-iterations 5 --rake-damping .5 --weight-clip 20 \
  --temperature .50 --tail-mix .10 --draws 128 --consensus-strength "$STRENGTH" \
  "${PRIOR_ARGS[@]}" --seed "$SEED" --output "$OUTROOT/resampled"
AFTER_RESAMPLE=$(date +%s)
printf 'pre_eval_runtime_seconds=%s\n' "$((AFTER_RESAMPLE - START))" >> "$PROVENANCE"

"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$GRAPH" \
  --sample "B=$SOURCE/resampled/b_reference.csv" \
  --sample "G=$SOURCE/resampled/g_reference.csv" \
  --sample "R=$SOURCE/resampled/uniform_control.csv" \
  --sample "$VARIANT=$OUTROOT/resampled/selected.csv" \
  --seed "$SEED" --output "$OUTROOT/eval"
