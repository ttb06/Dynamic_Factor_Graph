#!/usr/bin/env bash
#SBATCH --job-name=llmtd-ext-prog-e
#SBATCH --partition=aicentre
#SBATCH --qos=unprivileged
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${DATASET:?}"; : "${GRAPH_JOB_ID:?}"; : "${CONTROLLER_JOB_ID:?}"
ROOT=/home/cs.aau.dk/tungkvt/binhtt2
REPO="$ROOT/baseline_repos/tabdiff"; A="$REPO/analysis/llm_feedback_20260922"
PY="$ROOT/baseline_runs/envs/grdm/bin/python"
sample="$A/outputs/extended_progressive_controller_${DATASET}_${CONTROLLER_JOB_ID}/samples"
graph="$A/outputs/extended_bscg_${DATASET}_${GRAPH_JOB_ID}/graph.json"
cd "$REPO"
"$PY" "$A/scripts/evaluate_shoppers_baseline_suite.py" --repo "$REPO" --dataset "$DATASET" --graph "$graph" \
  --sample "tabdiff=$sample/unconditional.csv" \
  --sample "final_repair=$sample/final_reward_calibrated.csv" \
  --sample "proposed=$sample/x0_reward_calibrated_then_final_reward.csv" \
  --seed 20261447 --output "$A/outputs/extended_progressive_eval_${DATASET}_${SLURM_JOB_ID}"
