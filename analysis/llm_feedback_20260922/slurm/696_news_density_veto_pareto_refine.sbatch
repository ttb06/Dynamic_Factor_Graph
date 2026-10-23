#!/usr/bin/env bash
# Focused refinement around the no-simplex density-veto Pareto point.
#SBATCH --job-name=llmtd-news-veto-refine
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

case "$i" in
  0) LABEL=o15_f15_s0;   O=.15; F=.15; SLACK=0;;
  1) LABEL=o20_f15_s0;   O=.20; F=.15; SLACK=0;;
  2) LABEL=o25_f10_s0;   O=.25; F=.10; SLACK=0;;
  3) LABEL=o15_f25_s0;   O=.15; F=.25; SLACK=0;;
  4) LABEL=o10_f25_s0;   O=.10; F=.25; SLACK=0;;
  5) LABEL=o20_f20_s02;  O=.20; F=.20; SLACK=.02;;
esac

WEIGHTS="documented_summary_order=$O,documented_functional_dependency=$F,documented_simplex=0,documented_conditional_simplex=0"
OUT="$A/outputs/news_veto_refine_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_documented_semantic_graph_20261016/graph.json" \
  --family-strengths "$WEIGHTS" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode residual --strength 0 --passes 1 --trees 96 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 1 --projection-min-t 0 --projection-max-t .25 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-density-veto --projection-density-slack "$SLACK" \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "veto_refined=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
