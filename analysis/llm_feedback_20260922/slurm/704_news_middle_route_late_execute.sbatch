#!/usr/bin/env bash
# Freeze a per-row program mask/target from one middle state, then execute the
# frozen route only in the last 15% of learned log-sigma.  This separates
# trajectory-dependent graph construction from safe late actuation.
#SBATCH --job-name=llmtd-news-route-late
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
  0) LABEL=r25_q0005; ROUTE=.25; THRESHOLD=.0005;;
  1) LABEL=r35_q0005; ROUTE=.35; THRESHOLD=.0005;;
  2) LABEL=r45_q0005; ROUTE=.45; THRESHOLD=.0005;;
  3) LABEL=r25_q001;  ROUTE=.25; THRESHOLD=.001;;
  4) LABEL=r35_q001;  ROUTE=.35; THRESHOLD=.001;;
  5) LABEL=r45_q001;  ROUTE=.45; THRESHOLD=.001;;
esac

OUT="$A/outputs/news_route_late_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_documented_semantic_graph_20261016/graph.json" \
  --include-families documented_summary_order \
  --family-strengths documented_summary_order=.25 \
  --program-transport rank \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode residual --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 1 --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-t "$ROUTE" --projection-stability-mode rank \
  --projection-stability-threshold "$THRESHOLD" \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "middle_route_late_execute=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
