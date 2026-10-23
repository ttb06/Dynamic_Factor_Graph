#!/usr/bin/env bash
# Low-distortion hybrid sweep.  The verified LLM projections remain at full
# strength; statistical-order projection and satisfied-edge messages are kept
# deliberately small to seek Trend/Beta gains without the R2 loss of the
# earlier order=.25, factor=.05 hybrid.
#SBATCH --job-name=llmtd-news-hybrid-low
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
  0) LABEL=order005_factor000; ORDER=.05; FACTOR=0;;
  1) LABEL=order010_factor000; ORDER=.10; FACTOR=0;;
  2) LABEL=order010_factor002; ORDER=.10; FACTOR=.02;;
  3) LABEL=order015_factor002; ORDER=.15; FACTOR=.02;;
esac

SEED=20261607
OUT="$A/outputs/news_hybrid_low_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --graph "$GRAPH" \
  --include-families statistical_order_program,llm_directional_bound \
  --factor-usage-roles actuate --factor-families llm_directional_bound \
  --family-strengths "statistical_order_program=$ORDER,llm_directional_bound=1" \
  --program-transport rank \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed "$SEED" \
  --mode copula --strength "$FACTOR" --passes 1 --trees 128 --min-r2 .05 --clip 1 \
  --factor-residual-gate .5 --factor-uncertainty-gate 1 \
  --factor-uncertainty-trees 16 \
  --factor-min-t .15 --factor-max-t .65 --factor-schedule middle_peak \
  --interval-space log_sigma_quantile \
  --projection-strength 1 --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-t .35 --projection-stability-mode rank \
  --projection-stability-threshold .001 --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "hybrid_low=$OUT/samples/dense_factor.csv" \
  --seed "$SEED" --output "$OUT/eval"
