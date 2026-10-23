#!/usr/bin/env bash
# Each constraint edge is gated by the learned noise levels of its own endpoint
# columns, instead of one global median sigma clock.
#SBATCH --job-name=llmtd-news-edge-clock
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
i=$SLURM_ARRAY_TASK_ID
case "$i" in
  0) LABEL=edge_middle;        LO=.20; HI=.60; SCHEDULE=flat;;
  1) LABEL=edge_narrow;        LO=.30; HI=.55; SCHEDULE=flat;;
  2) LABEL=edge_middle_smooth; LO=.20; HI=.60; SCHEDULE=middle_peak;;
esac

OUT="$A/outputs/news_edge_noise_clock_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261603 \
  --mode copula --strength 0 --passes 1 \
  --trees 192 --min-r2 .02 --clip 2.0 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 1 \
  --projection-interval-space edge_log_sigma_quantile \
  --projection-min-t "$LO" --projection-max-t "$HI" \
  --projection-schedule "$SCHEDULE" \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "final_repair=$OUT/samples/final_repair.csv" \
  --sample "edge_local_projection=$OUT/samples/dense_factor.csv" \
  --sample "edge_local_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed 20261603 --output "$OUT/eval"
