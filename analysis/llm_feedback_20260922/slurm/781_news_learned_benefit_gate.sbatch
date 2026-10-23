#!/usr/bin/env bash
# Learned row x edge x time activation gate, fitted only on forward-noised train rows.
#SBATCH --job-name=llmtd-news-learned-gate
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5%6
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
GATE="$A/outputs/news_edge_benefit_time_1097951/benefit_gate/manifest.json"
i=$SLURM_ARRAY_TASK_ID

MODEL_IDS=typed_004,typed_005,typed_007,typed_008,typed_009,typed_010,typed_011,typed_012,typed_020
NAME=unknown; STRENGTH=.5; TMIN=.03; TMAX=.95; STAGE=predictor; TRUST=.5; IDS="$MODEL_IDS"
case "$i" in
  0) NAME=gate_full_predictor_w05;;
  1) NAME=gate_full_both_w05; STAGE=both;;
  2) NAME=gate_full_predictor_w025; STRENGTH=.25;;
  3) NAME=gate_full_predictor_w10; STRENGTH=1;;
  4) NAME=gate_mid_predictor_w05; TMIN=.20; TMAX=.70;;
  5) NAME=gate_good_predictor_w05; IDS=typed_004,typed_007,typed_008,typed_009,typed_012,typed_020;;
esac
OUT="$A/outputs/news_learned_gate_${NAME}_${SLURM_ARRAY_JOB_ID}_${i}"
EDGE_ARGS=()
if [[ -n "$IDS" ]]; then EDGE_ARGS+=(--include-edge-ids "$IDS"); fi

cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --regression-target-mode numerical \
  --graph "$GRAPH" "${EDGE_ARGS[@]}" \
  --config "$TRAIN/config.pkl" --checkpoint "$TRAIN/ema_model_epoch_4000.pt" \
  --samples 8192 --steps 50 --seed 20261024 \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --family-strengths llm_directional_bound=1,statistical_order_program=1,statistical_functional_program=1,statistical_sum_program=1,statistical_conditional_sum_program=1 \
  --projection-strength "$STRENGTH" --projection-min-t "$TMIN" --projection-max-t "$TMAX" \
  --projection-interval-space time --projection-schedule flat \
  --projection-route-mode fixed --projection-route-t -1 \
  --projection-benefit-gate "$GATE" \
  --projection-trust-radius "$TRUST" --guidance-stage "$STAGE" \
  --stochastic-sampler --second-order-correction --output "$OUT/sample"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "learned_${NAME}=$OUT/sample/samples/dense_factor.csv" \
  --seed 20261024 --skip-detection --output "$OUT/eval"
