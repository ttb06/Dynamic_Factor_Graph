#!/usr/bin/env bash
# Three-seed paired confirmation for the two News operating points selected
# before this batch.  No test-driven hyperparameter sweep is performed here.
#SBATCH --job-name=llmtd-news-qconfirm
#SBATCH --partition=prioritized
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID

if (( i < 3 )); then
  VARIANT=one_shot_q4_s010
  LOCAL=$i
  PULSE=(--distribution-transport-strength .10 --distribution-transport-min-t .20 --distribution-transport-max-t .35 --distribution-transport-once)
else
  VARIANT=two_pulse_q4_s005
  LOCAL=$((i - 3))
  PULSE=(--distribution-transport-strength .05 --distribution-transport-min-t .10 --distribution-transport-max-t .40 --distribution-transport-max-calls 2 --distribution-transport-min-call-gap .18)
fi
SEED=$((20261200 + LOCAL))
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
OUT="$A/outputs/news_quantile_confirm_${VARIANT}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"

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
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40 \
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak \
  --projection-route-mode auto_edge_valley_isotonic \
  --projection-stability-mode rank --projection-stability-threshold .001 \
  --projection-persistence-steps 2 --projection-trust-radius .75 \
  --distribution-transport-mode copula --distribution-transport-scope all \
  --distribution-transport-condition-target-quantiles 4 \
  --distribution-transport-schedule flat "${PULSE[@]}" \
  --categorical-guidance-mode none \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --output "$OUT/sample"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "Backbone=$OUT/sample/samples/unconditional.csv" \
  --sample "DynamicGraph=$OUT/sample/samples/dense_factor.csv" \
  --seed "$SEED" --skip-detection --output "$OUT/eval"
