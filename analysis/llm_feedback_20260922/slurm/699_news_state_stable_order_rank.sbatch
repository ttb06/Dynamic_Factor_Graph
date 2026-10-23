#!/usr/bin/env bash
# Row- and timestep-dependent activation for documented order hyperedges.
# A row-edge becomes active only after its proposed clean-state target is
# stable across consecutive reverse states; rank transport preserves marginals.
#SBATCH --job-name=llmtd-news-stable-rank
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

STRENGTH=.25
THRESHOLD=.25
MAX_T=.25
case "$i" in
  0) LABEL=q10_s25; THRESHOLD=.10;;
  1) LABEL=q25_s25; THRESHOLD=.25;;
  2) LABEL=q50_s25; THRESHOLD=.50;;
  3) LABEL=q100_s25; THRESHOLD=1.00;;
  4) LABEL=q25_s20; THRESHOLD=.25; STRENGTH=.20;;
  5) LABEL=q25_s25_late40; THRESHOLD=.25; MAX_T=.40;;
esac

OUT="$A/outputs/news_stable_rank_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
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
  --mode residual --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --projection-strength 1 --projection-min-t 0 --projection-max-t "$MAX_T" \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-stability-threshold "$THRESHOLD" \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "state_stable_order_rank=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
