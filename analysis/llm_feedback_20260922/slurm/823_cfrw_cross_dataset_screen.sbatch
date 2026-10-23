#!/usr/bin/env bash
# CFRW validation screen: News + Default x W0/W4/W8/W12.
# Each GPU task samples the locked non-adaptive G reference and one CFRW arm
# sequentially, verifies their unconditional hashes locally, and evaluates
# B/G/W* together.  All fitting, sampling and evaluation stay inside Slurm.
#SBATCH --job-name=llmtd-cfrw-screen
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
DRIVER="$A/scripts/dense_numeric_factor_messages_cfrw.py"

VARIANTS=(W0 W4 W8 W12)
i=$SLURM_ARRAY_TASK_ID
dataset_index=$((i / 4))
variant_index=$((i % 4))
VARIANT=${VARIANTS[$variant_index]}
SAMPLES=8192
STEPS=50

REGRESSION=()
FAMILY=()
EDGE_ARGS=()
if (( dataset_index == 0 )); then
  DATASET=news
  SEED=20261340
  GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
  CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
  CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"
  REGRESSION=(--regression-target-mode numerical)
  FAMILY=(
    --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5
    --family-transports statistical_order_program=rank
  )
elif (( dataset_index == 1 )); then
  DATASET=default
  SEED=20261341
  GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
  CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
  EDGE_ARGS=(--include-edge-ids old::r1_001,old::r1_014,old::r1_015,old::r1_017)
else
  echo "invalid dataset index: $dataset_index" >&2
  exit 64
fi

OUTROOT="$A/outputs/cfrw_screen_${DATASET}_${VARIANT}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"

{
  printf 'stage=CFRW_cross_dataset_validation\n'
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
  printf 'array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=%s\n' "$DATASET"
  printf 'variant=%s\n' "$VARIANT"
  printf 'seed=%s\n' "$SEED"
  printf 'samples=%s\n' "$SAMPLES"
  printf 'steps=%s\n' "$STEPS"
  printf 'host=%s\n' "$(hostname)"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_diff_sha256='; git diff --binary | sha256sum | cut -d' ' -f1
  printf 'checkpoint='; stat -c '%n size=%s mtime=%y' "$CHECKPOINT"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
    --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum \
    "$DRIVER" \
    "$A/scripts/factorized_residual_witness.py" \
    "$A/scripts/dense_numeric_factor_messages_arr.py" \
    "$GRAPH" "$CONFIG"
} > "$PROVENANCE"
"$PY" -VV > "$OUTROOT/python_version.txt" 2>&1
if ! "$PY" -m pip freeze > "$OUTROOT/environment_freeze.txt"; then
  printf 'pip_freeze_unavailable\n' > "$OUTROOT/environment_freeze.txt"
fi

COMMON=(
  --repo "$REPO" --dataset "$DATASET" "${REGRESSION[@]}"
  --graph "$GRAPH" "${EDGE_ARGS[@]}" --config "$CONFIG" --checkpoint "$CHECKPOINT"
  --samples "$SAMPLES" --steps "$STEPS" --seed "$SEED"
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat "${FAMILY[@]}"
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak
  --projection-route-mode auto_edge_valley_isotonic
  --projection-stability-mode rank --projection-stability-threshold .001
  --projection-persistence-steps 2 --projection-trust-radius .75
  --guidance-stage both --stochastic-sampler --second-order-correction
)

if [[ "$DATASET" == news ]]; then
  # Locked News PS-DFR q4 one-shot comparator from R2.
  REFERENCE=(
    --distribution-transport-mode copula --distribution-transport-scope all
    --distribution-transport-soft-quantiles
    --distribution-transport-soft-temperature .75
    --distribution-transport-condition-target-quantiles 4
    --distribution-transport-strength .10
    --distribution-transport-min-t .20 --distribution-transport-max-t .35
    --distribution-transport-schedule flat --distribution-transport-once
    --categorical-guidance-mode none
  )
else
  # Locked Default categorical/numerical hybrid from job 1099422 task 11.
  REFERENCE=(
    --distribution-transport-mode copula --distribution-transport-scope all
    --distribution-transport-strength .50
    --distribution-transport-condition-target
    --distribution-transport-preserve-tail .10
    --distribution-transport-min-t .10 --distribution-transport-max-t .40
    --distribution-transport-schedule flat --distribution-transport-once
    --categorical-guidance-mode marginal_target
    --categorical-guidance-strength .25
    --categorical-guidance-min-t .20 --categorical-guidance-max-t .50
    --categorical-guidance-schedule middle_peak
    --categorical-guidance-marginal-scale .10
    --categorical-guidance-pair-scale 2
  )
fi

case "$VARIANT" in
  W0)
    CANDIDATE=(
      --residual-witness-mode marginal
      --residual-witness-trees 32 --residual-witness-min-r2 .02
      --residual-witness-max-pair-heads 0
      --residual-witness-top-factors-per-row 1
      --residual-witness-top-rows-per-factor .15
      --residual-witness-min-score .25 --residual-witness-clip 2
      --distribution-transport-mode none --distribution-transport-scope all
      --distribution-transport-strength .10
      --distribution-transport-min-t .20 --distribution-transport-max-t .35
      --distribution-transport-schedule flat --distribution-transport-once
      --categorical-guidance-mode none
    )
    ;;
  W4|W8|W12)
    MAX_PAIR_HEADS=${VARIANT#W}
    CANDIDATE=(
      --residual-witness-mode marginal_pair
      --residual-witness-trees 32 --residual-witness-min-r2 .02
      --residual-witness-max-pair-heads "$MAX_PAIR_HEADS"
      --residual-witness-top-factors-per-row 1
      --residual-witness-top-rows-per-factor .15
      --residual-witness-min-score .25 --residual-witness-clip 2
      --distribution-transport-mode none --distribution-transport-scope all
      --distribution-transport-strength .10
      --distribution-transport-min-t .20 --distribution-transport-max-t .35
      --distribution-transport-schedule flat --distribution-transport-once
      --categorical-guidance-mode none
    )
    ;;
  *)
    echo "unknown CFRW arm: $VARIANT" >&2
    exit 64
    ;;
esac

REFERENCE_COMMAND=("$PY" -u "$DRIVER" "${COMMON[@]}" "${REFERENCE[@]}" --output "$OUTROOT/reference")
CANDIDATE_COMMAND=("$PY" -u "$DRIVER" "${COMMON[@]}" "${CANDIDATE[@]}" --output "$OUTROOT/candidate")
{
  printf 'reference_command='
  printf '%q ' "${REFERENCE_COMMAND[@]}"
  printf '\n'
  printf 'candidate_command='
  printf '%q ' "${CANDIDATE_COMMAND[@]}"
  printf '\n'
} > "$OUTROOT/command.txt"
"${REFERENCE_COMMAND[@]}"
"${CANDIDATE_COMMAND[@]}"

sha256sum \
  "$OUTROOT/reference/samples/unconditional.csv" \
  "$OUTROOT/candidate/samples/unconditional.csv" \
  > "$OUTROOT/unconditional_sha256.txt"
hash_count=$(cut -d' ' -f1 "$OUTROOT/unconditional_sha256.txt" | sort -u | wc -l)
if [[ "$hash_count" -ne 1 ]]; then
  echo "paired reference/candidate unconditional samples differ" >&2
  exit 65
fi

EVAL_SAMPLES=(
  --sample "B=$OUTROOT/reference/samples/unconditional.csv"
  --sample "G=$OUTROOT/reference/samples/dense_factor.csv"
  --sample "$VARIANT=$OUTROOT/candidate/samples/dense_factor.csv"
)
if [[ "$DATASET" == news ]]; then
  "$PY" -u "$A/scripts/evaluate_news_public_constraints.py" \
    --repo "$REPO" "${EVAL_SAMPLES[@]}" \
    --seed "$SEED" --skip-detection --output "$OUTROOT/eval"
else
  "$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" \
    --repo "$REPO" --dataset "$DATASET" --graph "$GRAPH" \
    "${EVAL_SAMPLES[@]}" --seed "$SEED" --max-rows "$SAMPLES" \
    --output "$OUTROOT/eval"
fi
