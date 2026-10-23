#!/usr/bin/env bash
# Train-calibrated edge and interval selection; no heldout/test metric enters selection.
#SBATCH --job-name=llmtd-news-benefit-route
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
TRAIN="$A/outputs/full_train_news_official_regression_1093974"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
i=$SLURM_ARRAY_TASK_ID

GOOD=typed_004,typed_007,typed_008,typed_009,typed_012,typed_020
WITHOUT_LDA=typed_003,typed_004,typed_005,typed_007,typed_008,typed_009,typed_010,typed_012,typed_020,typed_022
NAME=unknown; IDS="$GOOD"; STRENGTH=.5; TMIN=.25; TMAX=.65; SCHEDULE=flat; ROUTE=.55
case "$i" in
  0) NAME=good_fixed55_w05;;
  1) NAME=good_fixed55_w10; STRENGTH=1;;
  2) NAME=good_peak_w05; TMIN=.35; SCHEDULE=middle_peak; ROUTE=-1;;
  3) NAME=good_live_w05; ROUTE=-1;;
  4) NAME=without_lda_live_w05; IDS="$WITHOUT_LDA"; ROUTE=-1;;
  5) NAME=good_no_llm_fixed55; IDS=typed_004,typed_007,typed_008,typed_009,typed_012;;
esac
OUT="$A/outputs/news_benefit_route_${NAME}_${SLURM_ARRAY_JOB_ID}_${i}"

cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --graph "$GRAPH" --include-edge-ids "$IDS" \
  --config "$TRAIN/config.pkl" --checkpoint "$TRAIN/ema_model_epoch_4000.pt" \
  --samples 8192 --steps 50 --seed 20261024 \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 \
  --family-transports statistical_order_program=rank \
  --projection-strength "$STRENGTH" --projection-min-t "$TMIN" --projection-max-t "$TMAX" \
  --projection-interval-space time --projection-schedule "$SCHEDULE" \
  --projection-route-mode fixed --projection-route-t "$ROUTE" \
  --projection-stability-mode rank --projection-stability-threshold .001 \
  --projection-persistence-steps 2 --projection-trust-radius .50 \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --output "$OUT/sample"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "benefit_${NAME}=$OUT/sample/samples/dense_factor.csv" \
  --seed 20261024 --skip-detection --output "$OUT/eval"
