#!/usr/bin/env bash
# Robust automatic routing: isotonic regression pools high-noise spikes before
# locating the largest graph-readability transition.  Three paired seeds and
# two actuation strengths are sufficient to choose a main configuration.
#SBATCH --job-name=llmtd-news-auto-iso
#SBATCH --partition=prioritized
#SBATCH --array=0-5%6
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
  0) LABEL=s075; STRENGTH=.75; SEED=20261605;;
  1) LABEL=s075; STRENGTH=.75; SEED=20261606;;
  2) LABEL=s075; STRENGTH=.75; SEED=20261607;;
  3) LABEL=s100; STRENGTH=1.00; SEED=20261605;;
  4) LABEL=s100; STRENGTH=1.00; SEED=20261606;;
  5) LABEL=s100; STRENGTH=1.00; SEED=20261607;;
esac

OUT="$A/outputs/news_auto_isotonic_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}_s${SEED}"
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
  --projection-route-mode auto_isotonic --projection-stability-mode rank \
  --projection-stability-threshold .001 --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "final_repair=$OUT/samples/final_repair.csv" \
  --sample "auto_isotonic=$OUT/samples/dense_factor.csv" \
  --sample "auto_isotonic_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed "$SEED" --output "$OUT/eval"
