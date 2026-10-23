#!/usr/bin/env bash
# Test whether semantically filtered soft relations improve the documented
# middle-route/late-execute mechanism without ever actuating a soft edge.
#SBATCH --job-name=llmtd-news-soft-factor
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:a40:1
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
i=$SLURM_ARRAY_TASK_ID

case "$i" in
  0) LABEL=route_only; GRAPH=order_actuation_only; STRENGTH=0;;
  1) LABEL=consensus_s02; GRAPH=order_plus_consensus_soft; STRENGTH=.02;;
  2) LABEL=consensus_s05; GRAPH=order_plus_consensus_soft; STRENGTH=.05;;
  3) LABEL=consensus_s10; GRAPH=order_plus_consensus_soft; STRENGTH=.10;;
  4) LABEL=luna_s05; GRAPH=order_plus_luna_soft; STRENGTH=.05;;
  5) LABEL=unfiltered_s05; GRAPH=order_plus_unfiltered_stable; STRENGTH=.05;;
esac

OUT="$A/outputs/news_authority_soft_factor_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --graph "$G/${GRAPH}.json" \
  --factor-usage-roles soft_factor_only \
  --family-strengths statistical_order_program=.25 --program-transport rank \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode copula --strength "$STRENGTH" --passes 1 --trees 128 --min-r2 .05 --clip 1 \
  --factor-min-t .15 --factor-max-t .65 --factor-schedule middle_peak \
  --interval-space log_sigma_quantile \
  --projection-strength 1 --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-t .35 --projection-stability-mode rank \
  --projection-stability-threshold .001 --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "authority_soft_factor=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
