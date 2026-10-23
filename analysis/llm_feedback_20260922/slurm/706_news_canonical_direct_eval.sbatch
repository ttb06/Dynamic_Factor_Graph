#!/usr/bin/env bash
# Canonical paired evaluation with converged/reproducible LogisticDetection.
#SBATCH --job-name=llmtd-news-canonical-eval
#SBATCH --partition=cpu
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
i=$SLURM_ARRAY_TASK_ID

case "$i" in
  0) SEED=20261604; LATE="$A/outputs/news_order_rank_1091204_4_s25_late15"; R35="$A/outputs/news_route_late_1091262_4_r35_q001"; R25="$A/outputs/news_route_late_1091262_3_r25_q001";;
  1) SEED=20261605; LATE="$A/outputs/news_rank_confirm_1091228_2_s25_late15_seed05"; R35="$A/outputs/news_route_confirm_1091268_0_r35_seed05"; R25="$A/outputs/news_route_confirm_1091268_2_r25_seed05";;
  2) SEED=20261606; LATE="$A/outputs/news_rank_confirm_1091228_3_s25_late15_seed06"; R35="$A/outputs/news_route_confirm_1091268_1_r35_seed06"; R25="$A/outputs/news_route_confirm_1091268_3_r25_seed06";;
esac

OUT="$A/outputs/news_canonical_eval_${SLURM_ARRAY_JOB_ID}_${i}_seed${SEED}"
cd "$REPO"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$LATE/samples/unconditional.csv" \
  --sample "late_rank=$LATE/samples/dense_factor.csv" \
  --sample "route_q35=$R35/samples/dense_factor.csv" \
  --sample "route_q24=$R25/samples/dense_factor.csv" \
  --seed "$SEED" --output "$OUT"
