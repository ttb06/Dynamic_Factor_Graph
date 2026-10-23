#!/usr/bin/env bash
# Isolate executable edges proposed from external semantics by the LLM and
# independently matched to the public News constraint catalog.
#SBATCH --job-name=llmtd-news-llm-edge
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:a40:1
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
  0) LABEL=llm_only_s100; INCLUDE=llm_directional_bound; WEIGHTS=llm_directional_bound=1.0;;
  1) LABEL=order_plus_llm_s025; INCLUDE=statistical_order_program,llm_directional_bound; WEIGHTS=statistical_order_program=.25,llm_directional_bound=.25;;
  2) LABEL=order_plus_llm_s100; INCLUDE=statistical_order_program,llm_directional_bound; WEIGHTS=statistical_order_program=.25,llm_directional_bound=1.0;;
esac

OUT="$A/outputs/news_llm_edge_actuation_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --graph "$GRAPH" --include-families "$INCLUDE" \
  --family-strengths "$WEIGHTS" --program-transport rank \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --interval-space log_sigma_quantile \
  --projection-strength 1 --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-t .35 --projection-stability-mode rank \
  --projection-stability-threshold .001 --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "llm_edge_actuation=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
