#!/usr/bin/env bash
# Refine the near-Pareto order-rank mechanism on common fidelity/utility metrics.
# Rank transport preserves each acted column's empirical marginal and changes
# only its coupling to the other endpoints of the documented order triplet.
#SBATCH --job-name=llmtd-news-order-rank
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-7%6
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

STRENGTH=.25
MIN_T=0
MAX_T=.25
SCHEDULE=flat
INTERVAL=log_sigma_quantile
TREES=0
VETO_ARGS=()
case "$i" in
  0) LABEL=s10_late25; STRENGTH=.10;;
  1) LABEL=s15_late25; STRENGTH=.15;;
  2) LABEL=s20_late25; STRENGTH=.20;;
  3) LABEL=s30_late25; STRENGTH=.30;;
  4) LABEL=s25_late15; MAX_T=.15;;
  5) LABEL=s25_late40; MAX_T=.40;;
  6) LABEL=s25_midlate50; MAX_T=.50; SCHEDULE=middle_peak;;
  7) LABEL=s25_edgewise25; INTERVAL=edge_log_sigma_quantile;;
esac

OUT="$A/outputs/news_order_rank_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_documented_semantic_graph_20261016/graph.json" \
  --include-families documented_summary_order \
  --family-strengths "documented_summary_order=$STRENGTH" \
  --program-transport rank \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode residual --strength 0 --passes 1 --trees "$TREES" --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 1 --projection-min-t "$MIN_T" --projection-max-t "$MAX_T" \
  --projection-interval-space "$INTERVAL" --projection-schedule "$SCHEDULE" \
  "${VETO_ARGS[@]}" \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "order_rank=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
