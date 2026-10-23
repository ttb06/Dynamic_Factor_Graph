#!/usr/bin/env bash
# Soft graph is used only as a train-fitted action scorer. It can veto a
# documented rank action, but it never supplies a projection target itself.
#SBATCH --job-name=llmtd-news-soft-scorer
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
  0) LABEL=consensus_slack0;   GRAPH=order_plus_consensus_soft; SLACK=0;;
  1) LABEL=consensus_slack02;  GRAPH=order_plus_consensus_soft; SLACK=.02;;
  2) LABEL=luna_slack0;        GRAPH=order_plus_luna_soft;      SLACK=0;;
  3) LABEL=luna_slack02;       GRAPH=order_plus_luna_soft;      SLACK=.02;;
  4) LABEL=unfiltered_slack0;  GRAPH=order_plus_unfiltered_stable; SLACK=0;;
  5) LABEL=unfiltered_slack02; GRAPH=order_plus_unfiltered_stable; SLACK=.02;;
esac

OUT="$A/outputs/news_soft_action_scorer_${SLURM_ARRAY_JOB_ID}_${i}_${LABEL}"
cd "$REPO"
"$PY" "$A/scripts/dense_numeric_factor_messages.py" \
  --repo "$REPO" --dataset news --graph "$G/${GRAPH}.json" \
  --factor-usage-roles soft_factor_only \
  --family-strengths statistical_order_program=.25 --program-transport rank \
  --config "$A/outputs/full_train_news_1069382/config.pkl" \
  --checkpoint "$A/outputs/full_train_news_1069382/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --seed 20261604 \
  --mode copula --strength 0 --passes 1 --trees 128 --min-r2 .05 --clip 1 \
  --factor-min-t 0 --factor-max-t 0 --factor-schedule flat \
  --interval-space log_sigma_quantile \
  --projection-strength 1 --projection-min-t 0 --projection-max-t .15 \
  --projection-interval-space log_sigma_quantile --projection-schedule flat \
  --projection-route-t .35 --projection-stability-mode rank \
  --projection-stability-threshold .001 \
  --projection-density-veto --projection-density-slack "$SLACK" \
  --output "$OUT"

"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$OUT/samples/unconditional.csv" \
  --sample "soft_graph_action_scorer=$OUT/samples/dense_factor.csv" \
  --seed 20261604 --output "$OUT/eval"
