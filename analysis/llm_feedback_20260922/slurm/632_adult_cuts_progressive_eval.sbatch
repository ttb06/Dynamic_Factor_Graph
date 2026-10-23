#!/usr/bin/env bash
#SBATCH --job-name=llmtd-adult-cuts-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CONTROLLER_JOB_ID:?CONTROLLER_JOB_ID required}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
SAMPLE="$A/outputs/adult_cuts_progressive_controller_${CONTROLLER_JOB_ID}/samples"
cd "$REPO"
"$PY" "$A/scripts/evaluate_adult_cuts_constraints.py" --repo "$REPO" \
  --sample "tabdiff=$SAMPLE/unconditional.csv" \
  --sample "final_repair=$SAMPLE/final_reward_calibrated.csv" \
  --sample "proposed=$SAMPLE/x0_reward_calibrated_then_final_reward.csv" \
  --seed 20261447 --output "$A/outputs/adult_cuts_progressive_eval_${SLURM_JOB_ID}"
