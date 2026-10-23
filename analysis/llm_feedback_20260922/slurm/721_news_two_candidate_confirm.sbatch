#!/usr/bin/env bash
# Two additional paired seeds for the only two candidates that survived the
# single-seed fidelity/utility screen.  Seed 20261604 already exists.
#SBATCH --job-name=llmtd-news-finalists
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
G="$A/outputs/news_constraint_injection_variants_v2_20261017"
LLMG="$A/outputs/news_llm_typed_completion_v2_1091599/actuation_graph.json"
i=$SLURM_ARRAY_TASK_ID

case "$i" in
  0) METHOD=llm_only; SEED=20261605;;
  1) METHOD=llm_only; SEED=20261606;;
  2) METHOD=state_gate; SEED=20261605;;
  3) METHOD=state_gate; SEED=20261606;;
esac

OUT="$A/outputs/news_finalist_${METHOD}_s${SEED}_${SLURM_ARRAY_JOB_ID}_${i}"
cd "$REPO"
if [[ "$METHOD" == llm_only ]]; then
  "$PY" "$A/scripts/dense_numeric_factor_messages.py" \
    --repo "$REPO" --dataset news --graph "$LLMG" \
    --include-families llm_directional_bound \
    --family-strengths llm_directional_bound=1.0 --program-transport rank \
    --config "$A/outputs/full_train_news_1069382/config.pkl" \
    --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
    --samples 4096 --steps 50 --seed "$SEED" \
    --mode copula --strength 0 --passes 1 --trees 0 --min-r2 .05 --clip 1 \
    --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
    --interval-space log_sigma_quantile \
    --projection-strength 1 --projection-min-t 0 --projection-max-t .15 \
    --projection-interval-space log_sigma_quantile --projection-schedule flat \
    --projection-route-t .35 --projection-stability-mode rank \
    --projection-stability-threshold .001 --output "$OUT"
else
  "$PY" "$A/scripts/dense_numeric_factor_messages.py" \
    --repo "$REPO" --dataset news --graph "$G/order_plus_consensus_soft.json" \
    --factor-usage-roles soft_factor_only \
    --family-strengths statistical_order_program=.25 --program-transport rank \
    --config "$A/outputs/full_train_news_1069382/config.pkl" \
    --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
    --samples 4096 --steps 50 --seed "$SEED" \
    --mode copula --strength .05 --passes 1 --trees 128 --min-r2 .05 --clip 1 \
    --factor-residual-gate 1 --factor-uncertainty-gate -1 \
    --factor-min-t .15 --factor-max-t .65 --factor-schedule middle_peak \
    --interval-space log_sigma_quantile \
    --projection-strength 1 --projection-min-t 0 --projection-max-t .15 \
    --projection-interval-space log_sigma_quantile --projection-schedule flat \
    --projection-route-t .35 --projection-stability-mode rank \
    --projection-stability-threshold .001 --output "$OUT"
fi

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "$METHOD=$OUT/samples/dense_factor.csv" \
  --seed "$SEED" --output "$OUT/eval"
