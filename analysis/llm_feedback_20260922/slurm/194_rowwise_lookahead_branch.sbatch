#!/usr/bin/env bash
#SBATCH --job-name=llmtd-mid-lookahead
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CURATE_JOB_ID:?CURATE_JOB_ID required}"; : "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
: "${SAMPLES:=4096}"; : "${TARGET_T:=0.18}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/rowwise_lookahead_branch.py" --repo "$REPO" \
  --graph "$A/outputs/rowwise_mechanism_graph_${CURATE_JOB_ID}/active_graph.json" \
  --schedule "$A/outputs/rowwise_mechanism_trajectory_${TRAJECTORY_JOB_ID}/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples "$SAMPLES" --steps 50 --seed 20261006 --target-t "$TARGET_T" \
  --min-support 12 --drift .10 \
  --output "$A/outputs/rowwise_lookahead_${SLURM_JOB_ID}"
