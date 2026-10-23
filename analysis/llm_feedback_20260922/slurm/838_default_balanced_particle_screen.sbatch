#!/usr/bin/env bash
# Default oversample-and-select screen for train-only balanced particles.
#SBATCH --job-name=llmtd-particle-screen
#SBATCH --partition=prioritized
#SBATCH --array=0-7%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
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
DRIVER="$A/scripts/dense_numeric_factor_messages_arr.py"
RESAMPLER="$A/scripts/balanced_particle_resampling.py"
GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"

VARIANTS=(uniform32 rake_marg35 rake_pair35 rake_pair60 ratio_logit50 ratio_xgb50 hybrid_logit50 hybrid_xgb15)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
MODE=uniform_search
TEMPERATURE=.50
TAIL=.10
MAX_PAIRS=16
case "$VARIANT" in
  uniform32) MODE=uniform_search ;;
  rake_marg35) MODE=rake_marginal; TEMPERATURE=.35; MAX_PAIRS=0 ;;
  rake_pair35) MODE=rake_pair; TEMPERATURE=.35 ;;
  rake_pair60) MODE=rake_pair; TEMPERATURE=.60 ;;
  ratio_logit50) MODE=ratio_logit ;;
  ratio_xgb50) MODE=ratio_xgb ;;
  hybrid_logit50) MODE=hybrid_logit ;;
  hybrid_xgb15) MODE=hybrid_xgb; TAIL=.15 ;;
  *) echo "unknown variant" >&2; exit 64 ;;
esac

SEED=20261380
POOL_SAMPLES=16384
SELECTED_SAMPLES=8192
STEPS=50
OUTROOT="$A/outputs/default_particle_${VARIANT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"

{
  printf 'stage=Default_balanced_particle_screen\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=default\nvariant=%s\nmode=%s\nseed=%s\npool_samples=%s\nselected_samples=%s\nsteps=%s\n' \
    "$VARIANT" "$MODE" "$SEED" "$POOL_SAMPLES" "$SELECTED_SAMPLES" "$STEPS"
  printf 'temperature=%s\ntail_mix=%s\nmax_pairs=%s\ndraws=32\n' "$TEMPERATURE" "$TAIL" "$MAX_PAIRS"
  printf 'host=%s\nstarted_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_diff_sha256='; git diff --binary | sha256sum | cut -d' ' -f1
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" "$RESAMPLER" "$GRAPH" "$CONFIG" "$CHECKPOINT"
} > "$PROVENANCE"

POOL_COMMAND=("$PY" -u "$DRIVER"
  --repo "$REPO" --dataset default --graph "$GRAPH"
  --include-edge-ids old::r1_001,old::r1_014,old::r1_015,old::r1_017
  --config "$CONFIG" --checkpoint "$CHECKPOINT"
  --samples "$POOL_SAMPLES" --steps "$STEPS" --seed "$SEED"
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak
  --projection-route-mode auto_edge_valley_isotonic
  --projection-stability-mode rank --projection-stability-threshold .001
  --projection-persistence-steps 2 --projection-trust-radius .75
  --guidance-stage both --stochastic-sampler --second-order-correction
  --distribution-transport-mode copula --distribution-transport-scope all
  --distribution-transport-strength .50 --distribution-transport-condition-target
  --distribution-transport-preserve-tail .10
  --distribution-transport-min-t .10 --distribution-transport-max-t .40
  --distribution-transport-schedule flat --distribution-transport-once
  --categorical-guidance-mode marginal_target --categorical-guidance-strength .25
  --categorical-guidance-min-t .20 --categorical-guidance-max-t .50
  --categorical-guidance-schedule middle_peak
  --categorical-guidance-marginal-scale .10 --categorical-guidance-pair-scale 2
  --output "$OUTROOT/pool")

RESAMPLE_COMMAND=("$PY" -u "$RESAMPLER" --repo "$REPO" --dataset default
  --pool "$OUTROOT/pool/samples/dense_factor.csv"
  --backbone-pool "$OUTROOT/pool/samples/unconditional.csv"
  --count "$SELECTED_SAMPLES" --mode "$MODE"
  --bins 16 --pair-bins 4 --max-pairs "$MAX_PAIRS"
  --rake-iterations 5 --rake-damping .5 --weight-clip 20
  --temperature "$TEMPERATURE" --tail-mix "$TAIL" --draws 32
  --seed "$SEED" --output "$OUTROOT/resampled")

{
  printf 'pool_command='; printf '%q ' "${POOL_COMMAND[@]}"; printf '\n'
  printf 'resample_command='; printf '%q ' "${RESAMPLE_COMMAND[@]}"; printf '\n'
} > "$OUTROOT/command.txt"

START=$(date +%s)
"${POOL_COMMAND[@]}"
AFTER_POOL=$(date +%s)
"${RESAMPLE_COMMAND[@]}"
AFTER_RESAMPLE=$(date +%s)
{
  printf 'pool_runtime_seconds=%s\n' "$((AFTER_POOL - START))"
  printf 'resampling_runtime_seconds=%s\n' "$((AFTER_RESAMPLE - AFTER_POOL))"
  printf 'pre_eval_runtime_seconds=%s\n' "$((AFTER_RESAMPLE - START))"
} >> "$PROVENANCE"

"$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
  --dataset default --graph "$GRAPH" \
  --sample "B=$OUTROOT/resampled/b_reference.csv" \
  --sample "G=$OUTROOT/resampled/g_reference.csv" \
  --sample "R=$OUTROOT/resampled/uniform_control.csv" \
  --sample "$VARIANT=$OUTROOT/resampled/selected.csv" \
  --seed "$SEED" --output "$OUTROOT/eval"
