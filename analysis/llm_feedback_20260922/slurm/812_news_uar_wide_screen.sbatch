#!/usr/bin/env bash
# Six uncertainty-aware router arms x two paired seeds = twelve GPU tasks.
# Each task runs the locked PS-DFR reference and one candidate sequentially.
#SBATCH --job-name=llmtd-news-uar-r1
#SBATCH --partition=prioritized
#SBATCH --array=0-11%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
set +x

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DRIVER="$A/scripts/dense_numeric_factor_messages_uar.py"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"

VARIANTS=(
  sharp035
  confidence55_global025
  confidence55_abstain
  switchmargin15
  confidence55_switchmargin15
  confidence55_switchmargin15_tail05
)
i=$SLURM_ARRAY_TASK_ID
variant_index=$((i / 2))
replicate=$((i % 2))
VARIANT=${VARIANTS[$variant_index]}
SEED=$((20261250 + replicate))
OUTROOT="$A/outputs/news_uar_r1_${VARIANT}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
cd "$REPO"

{
  printf 'stage=R1_wide_screen\n'
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
  printf 'array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
  printf 'variant=%s\n' "$VARIANT"
  printf 'seed=%s\n' "$SEED"
  printf 'host=%s\n' "$(hostname)"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_diff_sha256='; git diff --binary | sha256sum | cut -d' ' -f1
  printf 'checkpoint='; stat -c '%n size=%s mtime=%y' "$CHECKPOINT"
  nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" \
    "$A/scripts/posterior_quantile_transport.py" \
    "$A/scripts/uncertainty_aware_quantile_transport.py" \
    "$GRAPH" "$CONFIG"
} > "$OUTROOT/provenance.txt"

COMMON=(
  --repo "$REPO" --dataset news --regression-target-mode numerical
  --graph "$GRAPH" --config "$CONFIG" --checkpoint "$CHECKPOINT"
  --samples 8192 --steps 50 --seed "$SEED"
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat
  --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5
  --family-transports statistical_order_program=rank
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40
  --projection-interval-space edge_log_sigma_quantile
  --projection-schedule middle_peak
  --projection-route-mode auto_edge_valley_isotonic
  --projection-stability-mode rank --projection-stability-threshold .001
  --projection-persistence-steps 2 --projection-trust-radius .75
  --distribution-transport-mode copula --distribution-transport-scope all
  --distribution-transport-schedule flat
  --distribution-transport-condition-target-quantiles 4
  --distribution-transport-soft-quantiles
  --distribution-transport-strength .05
  --distribution-transport-min-t .10 --distribution-transport-max-t .40
  --distribution-transport-max-calls 2
  --distribution-transport-min-call-gap .18
  --categorical-guidance-mode none
  --guidance-stage both --stochastic-sampler --second-order-correction
)

REFERENCE=(--distribution-transport-soft-temperature .75)
CANDIDATE=()
case "$VARIANT" in
  sharp035)
    CANDIDATE=(--distribution-transport-soft-temperature .35)
    ;;
  confidence55_global025)
    CANDIDATE=(
      --distribution-transport-soft-temperature .75
      --distribution-transport-confidence-floor .55
      --distribution-transport-confidence-full .75
      --distribution-transport-uncertainty-fallback global
      --distribution-transport-fallback-scale .25
    )
    ;;
  confidence55_abstain)
    CANDIDATE=(
      --distribution-transport-soft-temperature .75
      --distribution-transport-confidence-floor .55
      --distribution-transport-confidence-full .75
      --distribution-transport-uncertainty-fallback abstain
    )
    ;;
  switchmargin15)
    CANDIDATE=(
      --distribution-transport-soft-temperature .75
      --distribution-transport-switch-margin .15
    )
    ;;
  confidence55_switchmargin15)
    CANDIDATE=(
      --distribution-transport-soft-temperature .75
      --distribution-transport-confidence-floor .55
      --distribution-transport-confidence-full .75
      --distribution-transport-uncertainty-fallback global
      --distribution-transport-fallback-scale .25
      --distribution-transport-switch-margin .15
    )
    ;;
  confidence55_switchmargin15_tail05)
    CANDIDATE=(
      --distribution-transport-soft-temperature .75
      --distribution-transport-confidence-floor .55
      --distribution-transport-confidence-full .75
      --distribution-transport-uncertainty-fallback global
      --distribution-transport-fallback-scale .25
      --distribution-transport-switch-margin .15
      --distribution-transport-preserve-tail .05
    )
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 64
    ;;
esac

"$PY" -u "$DRIVER" "${COMMON[@]}" "${REFERENCE[@]}" \
  --output "$OUTROOT/reference"
"$PY" -u "$DRIVER" "${COMMON[@]}" "${CANDIDATE[@]}" \
  --output "$OUTROOT/candidate"

sha256sum \
  "$OUTROOT/reference/samples/unconditional.csv" \
  "$OUTROOT/candidate/samples/unconditional.csv" \
  > "$OUTROOT/unconditional_sha256.txt"
reference_hash=$(sha256sum "$OUTROOT/reference/samples/unconditional.csv" | cut -d' ' -f1)
candidate_hash=$(sha256sum "$OUTROOT/candidate/samples/unconditional.csv" | cut -d' ' -f1)
if [[ "$reference_hash" != "$candidate_hash" ]]; then
  echo "paired unconditional samples differ" >&2
  exit 65
fi

"$PY" -u "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "Backbone=$OUTROOT/reference/samples/unconditional.csv" \
  --sample "Reference=$OUTROOT/reference/samples/dense_factor.csv" \
  --sample "$VARIANT=$OUTROOT/candidate/samples/dense_factor.csv" \
  --seed "$SEED" --skip-detection --output "$OUTROOT/eval"

{
  printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'reference_unconditional_sha256=%s\n' "$reference_hash"
} >> "$OUTROOT/provenance.txt"
