#!/usr/bin/env bash
# Paired confirmation of the two Pareto route points from job 1092100:
# a later-middle route (q=.35,s=.75) and an earlier-middle route (q=.50,s=1).
# Here q=1 is the noisy start and q=0 is the clean end.  This confirms which
# middle state provides the better routed target; it is not a late-only test.
#SBATCH --job-name=llmtd-news-soft-route
#SBATCH --partition=prioritized
#SBATCH --array=0-3%4
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
  0) LABEL=latermid_q035_s075; ROUTE=.35; STRENGTH=.75; SEED=20261605;;
  1) LABEL=latermid_q035_s075; ROUTE=.35; STRENGTH=.75; SEED=20261606;;
  2) LABEL=earlymid_q050_s100; ROUTE=.50; STRENGTH=1.00; SEED=20261605;;
  3) LABEL=earlymid_q050_s100; ROUTE=.50; STRENGTH=1.00; SEED=20261606;;
esac

OUT="$A/outputs/news_soft_route_confirm_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}_s${SEED}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --graph "$GRAPH" \
  --include-families llm_directional_bound \
  --family-strengths llm_directional_bound=1.0 --program-transport rank \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed "$SEED" \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --interval-space log_sigma_quantile \
  --projection-strength "$STRENGTH" --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-t "$ROUTE" --projection-stability-mode rank \
  --projection-stability-threshold .001 --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "soft_route=$OUT/samples/dense_factor.csv" \
  --seed "$SEED" --output "$OUT/eval"
