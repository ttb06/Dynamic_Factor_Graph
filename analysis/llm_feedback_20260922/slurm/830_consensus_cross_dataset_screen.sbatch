#!/usr/bin/env bash
# Paired News+Default action-space consensus screen.
#SBATCH --job-name=llmtd-consensus
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
DRIVER="$A/scripts/dense_numeric_factor_messages_consensus.py"

VARIANTS=(aligned15 aligned30 pcgrad15 pcgrad30)
i=$SLURM_ARRAY_TASK_ID
dataset_index=$((i / 4))
variant_index=$((i % 4))
VARIANT=${VARIANTS[$variant_index]}
SAMPLES=8192
STEPS=50
REGRESSION=()
FAMILY=()
EDGE_ARGS=()
CATEGORICAL=(--categorical-guidance-mode none)

if (( dataset_index == 0 )); then
  DATASET=news
  SEED=20261360
  GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
  CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
  CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"
  REGRESSION=(--regression-target-mode numerical)
  FAMILY=(
    --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5
    --family-transports statistical_order_program=rank
  )
  TRANSPORT_STRENGTH=.10
  TRANSPORT_MIN_T=.20
  TRANSPORT_MAX_T=.35
elif (( dataset_index == 1 )); then
  DATASET=default
  SEED=20261361
  GRAPH="$A/outputs/actionable_union_eval_default_1060955_1/evaluation_graph.json"
  CONFIG="$A/outputs/full_train_default_1033760/config.pkl"
  CHECKPOINT="$A/outputs/full_train_default_1033760/ema_model_epoch_8000.pt"
  EDGE_ARGS=(--include-edge-ids old::r1_001,old::r1_014,old::r1_015,old::r1_017)
  CATEGORICAL=(
    --categorical-guidance-mode marginal_target
    --categorical-guidance-strength .25
    --categorical-guidance-min-t .20 --categorical-guidance-max-t .50
    --categorical-guidance-schedule middle_peak
    --categorical-guidance-marginal-scale .10
    --categorical-guidance-pair-scale 2
  )
  TRANSPORT_STRENGTH=.50
  TRANSPORT_MIN_T=.10
  TRANSPORT_MAX_T=.40
else
  echo "invalid dataset index" >&2
  exit 64
fi

MODE=${VARIANT%%[0-9]*}
RATIO=.15
if [[ "$VARIANT" == *30 ]]; then RATIO=.30; fi
OUTROOT="$A/outputs/consensus_${DATASET}_${VARIANT}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"

{
  printf 'stage=action_space_consensus_screen\n'
  printf 'job_id=%s\narray_job_id=%s\narray_task_id=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_JOB_ID" "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=%s\nvariant=%s\nmode=%s\nratio=%s\nseed=%s\nsamples=%s\nsteps=%s\n' \
    "$DATASET" "$VARIANT" "$MODE" "$RATIO" "$SEED" "$SAMPLES" "$STEPS"
  printf 'host=%s\nstarted_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_diff_sha256='; git diff --binary | sha256sum | cut -d' ' -f1
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" "$A/scripts/consensus_guidance_transport.py" \
    "$A/scripts/factorized_residual_witness.py" "$A/scripts/dense_numeric_factor_messages_arr.py" \
    "$GRAPH" "$CONFIG" "$CHECKPOINT"
} > "$PROVENANCE"

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
  "${CATEGORICAL[@]}"
)

if [[ "$DATASET" == news ]]; then
  REFERENCE=(
    --distribution-transport-mode copula --distribution-transport-scope all
    --distribution-transport-soft-quantiles --distribution-transport-soft-temperature .75
    --distribution-transport-condition-target-quantiles 4
    --distribution-transport-strength "$TRANSPORT_STRENGTH"
    --distribution-transport-min-t "$TRANSPORT_MIN_T" --distribution-transport-max-t "$TRANSPORT_MAX_T"
    --distribution-transport-schedule flat --distribution-transport-once
  )
else
  REFERENCE=(
    --distribution-transport-mode copula --distribution-transport-scope all
    --distribution-transport-condition-target --distribution-transport-preserve-tail .10
    --distribution-transport-strength "$TRANSPORT_STRENGTH"
    --distribution-transport-min-t "$TRANSPORT_MIN_T" --distribution-transport-max-t "$TRANSPORT_MAX_T"
    --distribution-transport-schedule flat --distribution-transport-once
  )
fi

CANDIDATE=(
  --consensus-mode "$MODE" --consensus-witness-ratio "$RATIO"
  --consensus-witness-trees 32 --consensus-witness-min-r2 .02
  --consensus-witness-pair-heads 4
  --consensus-witness-top-factors-per-row 1
  --consensus-witness-top-rows-per-factor .15
  --consensus-witness-min-score .25 --consensus-witness-clip 2
  --distribution-transport-mode none --distribution-transport-scope all
  --distribution-transport-strength "$TRANSPORT_STRENGTH"
  --distribution-transport-min-t "$TRANSPORT_MIN_T" --distribution-transport-max-t "$TRANSPORT_MAX_T"
  --distribution-transport-schedule flat --distribution-transport-once
)

REFERENCE_COMMAND=("$PY" -u "$DRIVER" "${COMMON[@]}" "${REFERENCE[@]}" --output "$OUTROOT/reference")
CANDIDATE_COMMAND=("$PY" -u "$DRIVER" "${COMMON[@]}" "${CANDIDATE[@]}" --output "$OUTROOT/candidate")
{
  printf 'reference_command='; printf '%q ' "${REFERENCE_COMMAND[@]}"; printf '\n'
  printf 'candidate_command='; printf '%q ' "${CANDIDATE_COMMAND[@]}"; printf '\n'
} > "$OUTROOT/command.txt"
"${REFERENCE_COMMAND[@]}"
"${CANDIDATE_COMMAND[@]}"

sha256sum "$OUTROOT/reference/samples/unconditional.csv" "$OUTROOT/candidate/samples/unconditional.csv" > "$OUTROOT/unconditional_sha256.txt"
if [[ "$(cut -d' ' -f1 "$OUTROOT/unconditional_sha256.txt" | sort -u | wc -l)" -ne 1 ]]; then
  echo "paired unconditional mismatch" >&2
  exit 65
fi

EVAL_SAMPLES=(
  --sample "B=$OUTROOT/reference/samples/unconditional.csv"
  --sample "G=$OUTROOT/reference/samples/dense_factor.csv"
  --sample "$VARIANT=$OUTROOT/candidate/samples/dense_factor.csv"
)
if [[ "$DATASET" == news ]]; then
  "$PY" -u "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
    "${EVAL_SAMPLES[@]}" --seed "$SEED" --skip-detection --output "$OUTROOT/eval"
else
  "$PY" -u "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" \
    --dataset "$DATASET" --graph "$GRAPH" "${EVAL_SAMPLES[@]}" \
    --seed "$SEED" --max-rows "$SAMPLES" --output "$OUTROOT/eval"
fi
