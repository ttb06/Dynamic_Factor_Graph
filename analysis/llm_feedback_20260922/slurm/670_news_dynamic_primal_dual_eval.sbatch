#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-primaldual-e
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${GUIDANCE_JOB_ID:?GUIDANCE_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
case "$SLURM_ARRAY_TASK_ID" in
  0) LABEL=conservative;;
  1) LABEL=responsive;;
esac
SAMPLE="$A/outputs/news_primal_dual_${GUIDANCE_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${LABEL}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SAMPLE/unconditional.csv" \
  --sample "final_repair=$SAMPLE/final_reward_calibrated.csv" \
  --sample "primal_dual_online=$SAMPLE/x0_reward_calibrated.csv" \
  --sample "primal_dual_then_certifier=$SAMPLE/x0_reward_calibrated_then_final_reward.csv" \
  --seed 20261447 --output "$A/outputs/news_primal_dual_eval_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${LABEL}"
