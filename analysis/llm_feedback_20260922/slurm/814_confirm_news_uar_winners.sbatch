#!/usr/bin/env bash
# Up to two selected UAR arms x six fresh paired seeds = twelve GPU tasks.
# Command-line --array may narrow this to 0-5 when only one arm is selected.
#SBATCH --job-name=llmtd-news-uar-confirm
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

: "${SELECTION_JSON:?submit with --export=ALL,SELECTION_JSON=<selection.json>}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
DRIVER="$A/scripts/dense_numeric_factor_messages_uar.py"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"

i=$SLURM_ARRAY_TASK_ID
winner_slot=$((i / 6))
seed_index=$((i % 6))
VARIANT=$("$PY" - "$SELECTION_JSON" "$winner_slot" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    winners = json.load(handle).get("winners", [])
index = int(sys.argv[2])
print(winners[index] if index < len(winners) else "")
PY
)
if [[ -z "$VARIANT" ]]; then
  echo "No winner in slot $winner_slot; exiting without generation."
  exit 0
fi

SEED=$((20261300 + seed_index))
OUTROOT="$A/outputs/news_uar_confirm_${VARIANT}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
cd "$REPO"

{
  printf 'stage=confirmation\n'
  printf 'selection_json=%s\n' "$SELECTION_JSON"
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
  printf 'array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
  printf 'winner_slot=%s\n' "$winner_slot"
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
    "$GRAPH" "$CONFIG" "$SELECTION_JSON"
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
  --distribution-transport-strength .05
  --distribution-transport-min-t .10 --distribution-transport-max-t .40
  --distribution-transport-max-calls 2
  --distribution-transport-min-call-gap .18
  --categorical-guidance-mode none
  --guidance-stage both --stochastic-sampler --second-order-correction
)

REFERENCE=(
  --distribution-transport-condition-target-quantiles 4
  --distribution-transport-soft-quantiles
  --distribution-transport-soft-temperature .75
)
FROZEN=(
  --distribution-transport-condition-target-quantiles 4
  --distribution-transport-soft-quantiles
  --distribution-transport-soft-temperature .75
  --distribution-transport-freeze-membership
)
GLOBAL=()
CANDIDATE=()
case "$VARIANT" in
  sharp035)
    CANDIDATE=(
      --distribution-transport-condition-target-quantiles 4
      --distribution-transport-soft-quantiles
      --distribution-transport-soft-temperature .35
    )
    ;;
  confidence55_global025)
    CANDIDATE=(
      --distribution-transport-condition-target-quantiles 4
      --distribution-transport-soft-quantiles
      --distribution-transport-soft-temperature .75
      --distribution-transport-confidence-floor .55
      --distribution-transport-confidence-full .75
      --distribution-transport-uncertainty-fallback global
      --distribution-transport-fallback-scale .25
    )
    ;;
  confidence55_abstain)
    CANDIDATE=(
      --distribution-transport-condition-target-quantiles 4
      --distribution-transport-soft-quantiles
      --distribution-transport-soft-temperature .75
      --distribution-transport-confidence-floor .55
      --distribution-transport-confidence-full .75
      --distribution-transport-uncertainty-fallback abstain
    )
    ;;
  switchmargin15)
    CANDIDATE=(
      --distribution-transport-condition-target-quantiles 4
      --distribution-transport-soft-quantiles
      --distribution-transport-soft-temperature .75
      --distribution-transport-switch-margin .15
    )
    ;;
  confidence55_switchmargin15)
    CANDIDATE=(
      --distribution-transport-condition-target-quantiles 4
      --distribution-transport-soft-quantiles
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
      --distribution-transport-condition-target-quantiles 4
      --distribution-transport-soft-quantiles
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
    echo "Unknown selected variant: $VARIANT" >&2
    exit 64
    ;;
esac

"$PY" -u "$DRIVER" "${COMMON[@]}" "${REFERENCE[@]}" \
  --output "$OUTROOT/reference"
"$PY" -u "$DRIVER" "${COMMON[@]}" "${GLOBAL[@]}" \
  --output "$OUTROOT/global"
"$PY" -u "$DRIVER" "${COMMON[@]}" "${FROZEN[@]}" \
  --output "$OUTROOT/frozen"
"$PY" -u "$DRIVER" "${COMMON[@]}" "${CANDIDATE[@]}" \
  --output "$OUTROOT/candidate"

sha256sum "$OUTROOT"/*/samples/unconditional.csv \
  > "$OUTROOT/unconditional_sha256.txt"
hash_count=$(cut -d' ' -f1 "$OUTROOT/unconditional_sha256.txt" | sort -u | wc -l)
if [[ "$hash_count" -ne 1 ]]; then
  echo "paired unconditional samples differ" >&2
  exit 65
fi

"$PY" -u "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "Backbone=$OUTROOT/reference/samples/unconditional.csv" \
  --sample "Global=$OUTROOT/global/samples/dense_factor.csv" \
  --sample "Frozen=$OUTROOT/frozen/samples/dense_factor.csv" \
  --sample "Reference=$OUTROOT/reference/samples/dense_factor.csv" \
  --sample "$VARIANT=$OUTROOT/candidate/samples/dense_factor.csv" \
  --seed "$SEED" --skip-detection --output "$OUTROOT/eval"

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" \
  >> "$OUTROOT/provenance.txt"
