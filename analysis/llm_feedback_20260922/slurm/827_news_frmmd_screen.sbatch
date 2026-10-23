#!/usr/bin/env bash
# News factorized RFF-MMD exploration: four paired GPU tasks.
#SBATCH --job-name=llmtd-news-frmmd
#SBATCH --partition=prioritized
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
DRIVER="$A/scripts/dense_numeric_factor_messages_frmmd.py"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"

VARIANTS=(mmd_marginal_one mmd_pair4_one mmd_pair8_one mmd_pair8_two)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SEED=20261350
SAMPLES=8192
STEPS=50
OUTROOT="$A/outputs/news_frmmd_${VARIANT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTROOT"
PROVENANCE="$OUTROOT/provenance.txt"
trap 'status=$?; printf "finished_at=%s\nexit_code=%s\n" "$(date --iso-8601=seconds)" "$status" >> "$PROVENANCE"; exit "$status"' EXIT
cd "$REPO"

{
  printf 'stage=News_factorized_RFF_MMD_screen\n'
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
  printf 'array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
  printf 'dataset=news\nvariant=%s\nseed=%s\nsamples=%s\nsteps=%s\n' \
    "$VARIANT" "$SEED" "$SAMPLES" "$STEPS"
  printf 'host=%s\nstarted_at=%s\n' "$(hostname)" "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_diff_sha256='; git diff --binary | sha256sum | cut -d' ' -f1
  printf 'checkpoint='; stat -c '%n size=%s mtime=%y' "$CHECKPOINT"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version \
    --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" "$A/scripts/factorized_rff_mmd.py" \
    "$A/scripts/dense_numeric_factor_messages_arr.py" "$GRAPH" "$CONFIG"
} > "$PROVENANCE"
"$PY" -VV > "$OUTROOT/python_version.txt" 2>&1

COMMON=(
  --repo "$REPO" --dataset news --regression-target-mode numerical
  --graph "$GRAPH" --config "$CONFIG" --checkpoint "$CHECKPOINT"
  --samples "$SAMPLES" --steps "$STEPS" --seed "$SEED"
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat
  --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5
  --family-transports statistical_order_program=rank
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak
  --projection-route-mode auto_edge_valley_isotonic
  --projection-stability-mode rank --projection-stability-threshold .001
  --projection-persistence-steps 2 --projection-trust-radius .75
  --guidance-stage both --stochastic-sampler --second-order-correction
)

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

MODE=marginal_pair
PAIR_HEADS=8
STRENGTH=.10
MIN_T=.20
MAX_T=.35
PULSE=(--distribution-transport-once)
case "$VARIANT" in
  mmd_marginal_one)
    MODE=marginal
    PAIR_HEADS=0
    ;;
  mmd_pair4_one)
    PAIR_HEADS=4
    ;;
  mmd_pair8_one)
    ;;
  mmd_pair8_two)
    STRENGTH=.05
    MIN_T=.10
    MAX_T=.40
    PULSE=(--distribution-transport-max-calls 2 --distribution-transport-min-call-gap .18)
    ;;
  *)
    echo "unknown FRMMD variant: $VARIANT" >&2
    exit 64
    ;;
esac

CANDIDATE=(
  --factor-mmd-mode "$MODE"
  --factor-mmd-features 64 --factor-mmd-bandwidth-scale 1
  --factor-mmd-max-pair-heads "$PAIR_HEADS"
  --factor-mmd-top-factors-per-row 1
  --factor-mmd-top-rows-per-factor .15
  --factor-mmd-min-score .25 --factor-mmd-action-clip .50
  --distribution-transport-mode none --distribution-transport-scope all
  --distribution-transport-strength "$STRENGTH"
  --distribution-transport-min-t "$MIN_T" --distribution-transport-max-t "$MAX_T"
  --distribution-transport-schedule flat "${PULSE[@]}"
  --categorical-guidance-mode none
)

REFERENCE_COMMAND=("$PY" -u "$DRIVER" "${COMMON[@]}" "${REFERENCE[@]}" --output "$OUTROOT/reference")
CANDIDATE_COMMAND=("$PY" -u "$DRIVER" "${COMMON[@]}" "${CANDIDATE[@]}" --output "$OUTROOT/candidate")
{
  printf 'reference_command='; printf '%q ' "${REFERENCE_COMMAND[@]}"; printf '\n'
  printf 'candidate_command='; printf '%q ' "${CANDIDATE_COMMAND[@]}"; printf '\n'
} > "$OUTROOT/command.txt"
"${REFERENCE_COMMAND[@]}"
"${CANDIDATE_COMMAND[@]}"

sha256sum \
  "$OUTROOT/reference/samples/unconditional.csv" \
  "$OUTROOT/candidate/samples/unconditional.csv" \
  > "$OUTROOT/unconditional_sha256.txt"
if [[ "$(cut -d' ' -f1 "$OUTROOT/unconditional_sha256.txt" | sort -u | wc -l)" -ne 1 ]]; then
  echo "paired reference/candidate unconditional samples differ" >&2
  exit 65
fi

"$PY" -u "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "B=$OUTROOT/reference/samples/unconditional.csv" \
  --sample "G=$OUTROOT/reference/samples/dense_factor.csv" \
  --sample "$VARIANT=$OUTROOT/candidate/samples/dense_factor.csv" \
  --seed "$SEED" --skip-detection --output "$OUTROOT/eval"
