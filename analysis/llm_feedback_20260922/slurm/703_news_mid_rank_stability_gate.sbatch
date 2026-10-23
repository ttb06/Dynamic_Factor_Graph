#!/usr/bin/env bash
# Open the graph in middle/late states, but actuate each row-edge only after
# its semantic target rank stabilizes across consecutive clean predictions.
#SBATCH --job-name=llmtd-news-mid-rank-gate
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

MAX_T=.50
SCHEDULE=flat
case "$i" in
  0) LABEL=q0005_flat50; THRESHOLD=.0005;;
  1) LABEL=q001_flat50;  THRESHOLD=.001;;
  2) LABEL=q0025_flat50; THRESHOLD=.0025;;
  3) LABEL=q005_flat50;  THRESHOLD=.005;;
  4) LABEL=q0025_peak65; THRESHOLD=.0025; MAX_T=.65; SCHEDULE=middle_peak;;
  5) LABEL=q0025_peak50; THRESHOLD=.0025; SCHEDULE=middle_peak;;
esac

OUT="$A/outputs/news_mid_rank_gate_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
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
  --projection-strength 1 --projection-min-t 0 --projection-max-t "$MAX_T" \
  --projection-interval-space log_sigma_quantile --projection-schedule "$SCHEDULE" \
  --projection-stability-mode rank --projection-stability-threshold "$THRESHOLD" \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "mid_rank_stability_gate=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
