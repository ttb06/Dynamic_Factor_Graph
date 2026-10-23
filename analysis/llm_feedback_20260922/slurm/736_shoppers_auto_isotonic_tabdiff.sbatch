#!/usr/bin/env bash
# Dataset transfer on the same TabDiff backbone and automatic router.
#SBATCH --job-name=llmtd-shop-auto-iso
#SBATCH --partition=prioritized
#SBATCH --array=0-1%2
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
GRAPH="$A/graphs/shoppers_documented_typed_graph.json"
i=$SLURM_ARRAY_TASK_ID
case "$i" in
  0) LABEL=s075; STRENGTH=.75;;
  1) LABEL=s100; STRENGTH=1.00;;
esac
OUT="$A/outputs/shoppers_auto_isotonic_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"

cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset shoppers --graph "$GRAPH" \
  --include-families documented_directional_bound \
  --family-strengths documented_directional_bound=1.0 --program-transport rank \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261607 \
  --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --interval-space log_sigma_quantile \
  --projection-strength "$STRENGTH" --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-mode auto_isotonic --projection-stability-mode rank \
  --projection-stability-threshold .001 --output "$OUT"

"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" \
  --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/actionable_union_eval_shoppers_1060955_0/evaluation_graph.json" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "auto_isotonic=$OUT/samples/dense_factor.csv" \
  --seed 20261607 --max-rows 4096 --output "$OUT/eval"
