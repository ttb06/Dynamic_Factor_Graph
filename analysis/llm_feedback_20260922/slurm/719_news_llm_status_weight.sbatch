#!/usr/bin/env bash
# Calibrate positive (already-satisfied) messages separately from corrective
# messages.  The current graph, seed, gate and trajectory are held fixed.
#SBATCH --job-name=llmtd-news-status-weight
#SBATCH --partition=prioritized
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
GRAPH="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
i=$SLURM_ARRAY_TASK_ID

case "$i" in
  0) LABEL=violation_only; SAT=0.00;;
  1) LABEL=positive_025;   SAT=0.25;;
  2) LABEL=positive_050;   SAT=0.50;;
esac

OUT="$A/outputs/news_llm_status_weight_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --graph "$GRAPH" \
  --include-families statistical_order_program,llm_directional_bound \
  --factor-usage-roles actuate --factor-families llm_directional_bound \
  --family-strengths statistical_order_program=.25,llm_directional_bound=0 \
  --program-transport rank \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode copula --strength .05 --passes 1 --trees 128 --min-r2 .05 --clip 1 \
  --factor-residual-gate .5 --factor-uncertainty-gate 1 \
  --factor-uncertainty-trees 16 \
  --factor-satisfied-strength-scale "$SAT" \
  --factor-violating-strength-scale 1 \
  --factor-min-t .15 --factor-max-t .65 --factor-schedule middle_peak \
  --interval-space log_sigma_quantile \
  --projection-strength 1 --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-t .35 --projection-stability-mode rank \
  --projection-stability-threshold .001 --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "status_weighted=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
