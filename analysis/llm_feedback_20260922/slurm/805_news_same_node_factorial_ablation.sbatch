#!/usr/bin/env bash
# Same-node/same-seed factorial ablation.  Running every cell sequentially on
# one GPU removes the hardware-dependent baseline drift seen across arrays.
#SBATCH --job-name=llmtd-news-factorial
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
OUTROOT="$A/outputs/news_same_node_factorial_${SLURM_JOB_ID}"
SEED=20261121

for VARIANT in typed_only conditional_only global_only full_conditional full_global; do
  PROJECTION=1
  TRANSPORT=(--distribution-transport-mode none)
  case "$VARIANT" in
    typed_only) ;;
    conditional_only)
      PROJECTION=0
      TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all --distribution-transport-strength .10 --distribution-transport-condition-target-quantiles 4 --distribution-transport-once)
      ;;
    global_only)
      PROJECTION=0
      TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all --distribution-transport-strength .10 --distribution-transport-once)
      ;;
    full_conditional)
      TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all --distribution-transport-strength .10 --distribution-transport-condition-target-quantiles 4 --distribution-transport-once)
      ;;
    full_global)
      TRANSPORT=(--distribution-transport-mode copula --distribution-transport-scope all --distribution-transport-strength .10 --distribution-transport-once)
      ;;
  esac
  cd "$REPO"
  "$PY" "$A/scripts/dense_numeric_factor_messages.py" \
    --repo "$REPO" --dataset news --regression-target-mode numerical \
    --graph "$GRAPH" \
    --config "$A/outputs/full_train_news_official_regression_1093974/config.pkl" \
    --checkpoint "$A/outputs/news_e2e_teacher_continue_full_1094488_0/train/ema_model_final.pt" \
    --samples 8192 --steps 50 --seed "$SEED" \
    --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
    --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
    --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 \
    --family-transports statistical_order_program=rank \
    --projection-strength "$PROJECTION" --projection-min-t .10 --projection-max-t .40 \
    --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak \
    --projection-route-mode auto_edge_valley_isotonic \
    --projection-stability-mode rank --projection-stability-threshold .001 \
    --projection-persistence-steps 2 --projection-trust-radius .75 \
    --distribution-transport-min-t .20 --distribution-transport-max-t .35 \
    --distribution-transport-schedule flat "${TRANSPORT[@]}" \
    --categorical-guidance-mode none \
    --guidance-stage both --stochastic-sampler --second-order-correction \
    --output "$OUTROOT/$VARIANT"
done

sha256sum "$OUTROOT"/*/samples/unconditional.csv
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "Backbone=$OUTROOT/typed_only/samples/unconditional.csv" \
  --sample "TypedGraph=$OUTROOT/typed_only/samples/dense_factor.csv" \
  --sample "ConditionalTransport=$OUTROOT/conditional_only/samples/dense_factor.csv" \
  --sample "GlobalTransport=$OUTROOT/global_only/samples/dense_factor.csv" \
  --sample "FullConditional=$OUTROOT/full_conditional/samples/dense_factor.csv" \
  --sample "FullGlobal=$OUTROOT/full_global/samples/dense_factor.csv" \
  --seed "$SEED" --skip-detection --output "$OUTROOT/eval"
