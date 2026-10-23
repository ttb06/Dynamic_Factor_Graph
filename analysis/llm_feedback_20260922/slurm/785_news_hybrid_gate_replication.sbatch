#!/usr/bin/env bash
# Same hybrid sampler, independently fitted train-noise benefit gate.
#SBATCH --job-name=llmtd-news-hybrid-repl
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
TRAIN="$A/outputs/full_train_news_official_regression_1093974"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
GATE="$A/outputs/news_edge_benefit_time_repl_1097982/benefit_gate/manifest.json"
OUT="$A/outputs/news_hybrid_gate_replication_${SLURM_JOB_ID}"

cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --graph "$GRAPH" --config "$TRAIN/config.pkl" \
  --checkpoint "$TRAIN/ema_model_epoch_4000.pt" \
  --samples 8192 --steps 50 --seed 20261024 \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --family-strengths llm_directional_bound=1,statistical_order_program=.15,statistical_functional_program=.5,statistical_sum_program=.5,statistical_conditional_sum_program=.5 \
  --family-transports statistical_order_program=rank \
  --projection-strength 1 --projection-min-t .10 --projection-max-t .40 \
  --projection-interval-space edge_log_sigma_quantile --projection-schedule middle_peak \
  --projection-route-mode auto_edge_valley_isotonic \
  --projection-stability-mode rank --projection-stability-threshold .001 \
  --projection-persistence-steps 2 --projection-trust-radius .50 \
  --projection-benefit-gate "$GATE" --benefit-gate-edge-ids typed_011,typed_012 \
  --guidance-stage both --stochastic-sampler --second-order-correction \
  --output "$OUT/sample"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "hybrid_gate_replication=$OUT/sample/samples/dense_factor.csv" \
  --seed 20261024 --skip-detection --output "$OUT/eval"
