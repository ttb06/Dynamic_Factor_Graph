#!/usr/bin/env bash
# Heun-consistent graph actuation at both predictor and corrector evaluations.
#SBATCH --job-name=llmtd-news-heun-guide
#SBATCH --partition=prioritized
#SBATCH --array=0-2%3
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
i=$SLURM_ARRAY_TASK_ID
case "$i" in
  0) LABEL=s050; STRENGTH=.50;;
  1) LABEL=s075; STRENGTH=.75;;
  2) LABEL=s100; STRENGTH=1.00;;
esac
SEED=20261609
OUT="$A/outputs/news_official_heun_guidance_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news \
  --graph "$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json" \
  --include-families llm_directional_bound --family-strengths llm_directional_bound=1 \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed "$SEED" --trees 0 --strength 0 \
  --factor-min-t 0 --factor-max-t 0 \
  --projection-strength "$STRENGTH" --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-mode auto_isotonic --projection-stability-mode rank \
  --projection-stability-threshold .001 \
  --stochastic-sampler --second-order-correction --guide-second-order \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff_official=$OUT/samples/unconditional.csv" \
  --sample "final_repair=$OUT/samples/final_repair.csv" \
  --sample "heun_guided=$OUT/samples/dense_factor.csv" \
  --sample "heun_guided_then_certifier=$OUT/samples/dense_factor_then_certifier.csv" \
  --seed "$SEED" --output "$OUT/eval"
