#!/usr/bin/env bash
# Two paired screening seeds. Every causal arm runs sequentially on the same
# node and uses the same checkpoint, seed, typed graph, pulse budget and trust
# geometry. Global -> frozen-soft isolates row conditioning; frozen -> dynamic
# isolates timestep reconfiguration.
#SBATCH --job-name=llmtd-news-psdfr
#SBATCH --partition=prioritized
#SBATCH --array=0-1%2
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
DRIVER="$A/scripts/dense_numeric_factor_messages_psdfr.py"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
CONFIG="$A/outputs/full_train_news_official_regression_1093974/config.pkl"
CHECKPOINT="$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt"
i=$SLURM_ARRAY_TASK_ID
SEED=$((20261240 + i))
OUTROOT="$A/outputs/news_psdfr_screen_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
mkdir -p "$OUTROOT"
cd "$REPO"

{
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
  printf 'array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
  printf 'seed=%s\n' "$SEED"
  printf 'host=%s\n' "$(hostname)"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_diff_sha256='; git diff --binary | sha256sum | cut -d' ' -f1
  printf 'checkpoint='; stat -c '%n size=%s mtime=%y' "$CHECKPOINT"
  nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader | sed 's/^/gpu=/'
  sha256sum "$DRIVER" "$A/scripts/posterior_quantile_transport.py" "$GRAPH" "$CONFIG"
} > "$OUTROOT/provenance.txt"

printf 'variant\trow_conditioned\tmembership_recomputed\tpulses\tstrength_per_pulse\n' > "$OUTROOT/variant_manifest.tsv"
printf 'hard_one_shot\t1\t0\t1\t0.10\n' >> "$OUTROOT/variant_manifest.tsv"
printf 'hard_two_pulse\t1\t1\t2\t0.05\n' >> "$OUTROOT/variant_manifest.tsv"
printf 'global_two_pulse\t0\t0\t2\t0.05\n' >> "$OUTROOT/variant_manifest.tsv"
printf 'frozen_soft_two_pulse\t1\t0\t2\t0.05\n' >> "$OUTROOT/variant_manifest.tsv"
printf 'dynamic_soft_two_pulse\t1\t1\t2\t0.05\n' >> "$OUTROOT/variant_manifest.tsv"

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
  --categorical-guidance-mode none
  --guidance-stage both --stochastic-sampler --second-order-correction
)

for VARIANT in \
  hard_one_shot \
  hard_two_pulse \
  global_two_pulse \
  frozen_soft_two_pulse \
  dynamic_soft_two_pulse
do
  case "$VARIANT" in
    hard_one_shot)
      TRANSPORT=(
        --distribution-transport-condition-target-quantiles 4
        --distribution-transport-strength .10
        --distribution-transport-min-t .20 --distribution-transport-max-t .35
        --distribution-transport-once
      )
      ;;
    hard_two_pulse)
      TRANSPORT=(
        --distribution-transport-condition-target-quantiles 4
        --distribution-transport-strength .05
        --distribution-transport-min-t .10 --distribution-transport-max-t .40
        --distribution-transport-max-calls 2
        --distribution-transport-min-call-gap .18
      )
      ;;
    global_two_pulse)
      TRANSPORT=(
        --distribution-transport-strength .05
        --distribution-transport-min-t .10 --distribution-transport-max-t .40
        --distribution-transport-max-calls 2
        --distribution-transport-min-call-gap .18
      )
      ;;
    frozen_soft_two_pulse)
      TRANSPORT=(
        --distribution-transport-condition-target-quantiles 4
        --distribution-transport-soft-quantiles
        --distribution-transport-soft-temperature .75
        --distribution-transport-freeze-membership
        --distribution-transport-strength .05
        --distribution-transport-min-t .10 --distribution-transport-max-t .40
        --distribution-transport-max-calls 2
        --distribution-transport-min-call-gap .18
      )
      ;;
    dynamic_soft_two_pulse)
      TRANSPORT=(
        --distribution-transport-condition-target-quantiles 4
        --distribution-transport-soft-quantiles
        --distribution-transport-soft-temperature .75
        --distribution-transport-strength .05
        --distribution-transport-min-t .10 --distribution-transport-max-t .40
        --distribution-transport-max-calls 2
        --distribution-transport-min-call-gap .18
      )
      ;;
  esac

  "$PY" -u "$DRIVER" "${COMMON[@]}" "${TRANSPORT[@]}" \
    --output "$OUTROOT/$VARIANT"
done

sha256sum "$OUTROOT"/*/samples/unconditional.csv > "$OUTROOT/unconditional_sha256.txt"

"$PY" -u "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "Backbone=$OUTROOT/hard_one_shot/samples/unconditional.csv" \
  --sample "HardOneShot=$OUTROOT/hard_one_shot/samples/dense_factor.csv" \
  --sample "HardTwoPulse=$OUTROOT/hard_two_pulse/samples/dense_factor.csv" \
  --sample "GlobalTwoPulse=$OUTROOT/global_two_pulse/samples/dense_factor.csv" \
  --sample "FrozenSoftTwoPulse=$OUTROOT/frozen_soft_two_pulse/samples/dense_factor.csv" \
  --sample "DynamicSoftTwoPulse=$OUTROOT/dynamic_soft_two_pulse/samples/dense_factor.csv" \
  --seed "$SEED" --skip-detection --output "$OUTROOT/eval"

date --iso-8601=seconds | sed 's/^/finished_at=/' >> "$OUTROOT/provenance.txt"
