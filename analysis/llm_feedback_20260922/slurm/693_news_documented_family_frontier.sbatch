#!/usr/bin/env bash
# Pareto frontier over documented relation families and weak projection.
#SBATCH --job-name=llmtd-news-doc-family
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
i=$SLURM_ARRAY_TASK_ID

FAMILIES=""
case "$i" in
  0) LABEL=all_s10;       STRENGTH=.10;;
  1) LABEL=all_s25;       STRENGTH=.25;;
  2) LABEL=function_s25;  STRENGTH=.25; FAMILIES=documented_functional_dependency;;
  3) LABEL=function_s50;  STRENGTH=.50; FAMILIES=documented_functional_dependency;;
  4) LABEL=simplex_s25;   STRENGTH=.25; FAMILIES=documented_simplex,documented_conditional_simplex;;
  5) LABEL=order_s25;     STRENGTH=.25; FAMILIES=documented_summary_order;;
esac

OUT="$A/outputs/news_documented_family_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
FAMILY_ARGS=()
if [[ -n "$FAMILIES" ]]; then FAMILY_ARGS=(--include-families "$FAMILIES"); fi
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_documented_semantic_graph_20261016/graph.json" \
  "${FAMILY_ARGS[@]}" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode residual --strength 0 --passes 1 --trees 64 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength "$STRENGTH" \
  --projection-interval-space log_sigma_quantile \
  --projection-min-t 0 --projection-max-t .25 --projection-schedule flat \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "documented_final_projection=$OUT/samples/final_repair.csv" \
  --sample "documented_trajectory=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
