#!/usr/bin/env bash
# Diagnose checkpoint selection before attributing a weak baseline to the
# graph mechanism.  Every epoch uses the same sampler, seed, graph and local
# evaluator; no held-out score is used to train the checkpoints.
#SBATCH --job-name=llmtd-news-ckpt-front
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
case "$i" in 0) EPOCH=4000;; 1) EPOCH=6000;; 2) EPOCH=8000;; esac
OUT="$A/outputs/news_checkpoint_frontier_${SLURM_ARRAY_JOB_ID}_${i}_e${EPOCH}"

cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --graph "$GRAPH" --config "$TRAIN/config.pkl" \
  --checkpoint "$TRAIN/ema_model_epoch_${EPOCH}.pt" \
  --samples 8192 --steps 50 --seed 20261024 \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 \
  --family-transports statistical_order_program=rank \
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40 \
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak \
  --projection-route-mode auto_edge_valley_isotonic --projection-stability-mode rank \
  --projection-stability-threshold .001 --projection-persistence-steps 2 \
  --projection-trust-radius .50 --guidance-stage both \
  --stochastic-sampler --second-order-correction --output "$OUT"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff_e${EPOCH}=$OUT/samples/unconditional.csv" \
  --sample "dynamic_e${EPOCH}=$OUT/samples/dense_factor.csv" \
  --seed 20261024 --skip-detection --output "$OUT/eval"

