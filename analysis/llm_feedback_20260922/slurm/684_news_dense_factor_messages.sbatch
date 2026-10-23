#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-factor
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
  0) LABEL=copula_s15_p2; MODE=copula; STRENGTH=.15; PASSES=2; PROJECTION=1.0;;
  1) LABEL=copula_s35_p3; MODE=copula; STRENGTH=.35; PASSES=3; PROJECTION=1.0;;
  2) LABEL=residual_s15_p2; MODE=residual; STRENGTH=.15; PASSES=2; PROJECTION=1.0;;
  3) LABEL=projection_only; MODE=copula; STRENGTH=0; PASSES=1; PROJECTION=1.0;;
  4) LABEL=copula_only_s35_p3; MODE=copula; STRENGTH=.35; PASSES=3; PROJECTION=0;;
  5) LABEL=copula_s60_p3; MODE=copula; STRENGTH=.60; PASSES=3; PROJECTION=1.0;;
esac

OUT="$A/outputs/news_dense_factor_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_public_typed_graph_20261015/graph.json" \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261601 \
  --mode "$MODE" --strength "$STRENGTH" --passes "$PASSES" \
  --trees 192 --min-r2 .02 --clip 2.0 \
  --factor-min-t .15 --factor-max-t .65 --factor-schedule middle_peak \
  --projection-strength "$PROJECTION" --projection-max-t .50 \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "final_repair=$OUT/samples/final_repair.csv" \
  --sample "dense_factor=$OUT/samples/dense_factor.csv" \
  --sample "dense_factor_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed 20261601 --output "$OUT/eval"
