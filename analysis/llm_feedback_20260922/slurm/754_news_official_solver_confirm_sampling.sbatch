#!/usr/bin/env bash
# Two-seed confirmation of the two Pareto-improving solver variants. Sampling
# only: evaluation runs on CPU so GPU slots return immediately.
#SBATCH --job-name=llmtd-news-off-confirm
#SBATCH --partition=prioritized
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"; i=$SLURM_ARRAY_TASK_ID
SEED=$((20261610 + i / 2))
if (( i % 2 == 0 )); then LABEL=predictor_s200; STRENGTH=2; EXTRA=(); else LABEL=heun_s100; STRENGTH=1; EXTRA=(--guide-second-order); fi
OUT="$A/outputs/news_official_confirm_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}_seed${SEED}"
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
  --projection-stability-threshold .001 --stochastic-sampler \
  --second-order-correction "${EXTRA[@]}" --output "$OUT"
