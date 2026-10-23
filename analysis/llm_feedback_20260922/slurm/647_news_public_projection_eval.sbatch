#!/usr/bin/env bash
#SBATCH --job-name=llmtd-news-project-e
#SBATCH --partition=cpu
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%A_%a.err
set -euo pipefail
: "${CONTROLLER_JOB_ID:?CONTROLLER_JOB_ID required}"

ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"
A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
labels=(balanced_projection balanced_lookahead)
i=$SLURM_ARRAY_TASK_ID
label=${labels[$i]}
sample="$A/outputs/news_projection_${label}_${CONTROLLER_JOB_ID}_${i}/samples"
output="$A/outputs/news_projection_eval_${SLURM_ARRAY_JOB_ID}_${i}"

cd "$REPO"
"$PY" "$A/scripts/evaluate_news_public_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$sample/unconditional.csv" \
  --sample "final_repair=$sample/final_reward_calibrated.csv" \
  --sample "proposed=$sample/x0_reward_calibrated_then_final_reward.csv" \
  --seed 20261447 --output "$output"
