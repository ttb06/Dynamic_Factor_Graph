#!/usr/bin/env bash
#SBATCH --job-name=llmtd-typed-guide
#SBATCH --partition=prioritized
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.out
#SBATCH --error=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff/analysis/llm_feedback_20260922/logs/%x_%j.err
set -euo pipefail
: "${CURATE_JOB_ID:?CURATE_JOB_ID required}"; : "${TRAJECTORY_JOB_ID:?TRAJECTORY_JOB_ID required}"
: "${GUIDANCE_STRENGTH:=1.0}"
REPO=/home/cs.aau.dk/tungkvt/binhtt2/baseline_repos/tabdiff
A="$REPO/analysis/llm_feedback_20260922"
PY=/home/cs.aau.dk/tungkvt/binhtt2/baseline_runs/envs/grdm/bin/python
cd "$REPO"
"$PY" "$A/scripts/typed_dynamic_graph_guidance.py" --repo "$REPO" --dataset shoppers \
  --graph "$A/outputs/typed_graph_curation_${CURATE_JOB_ID}/active_graph.json" \
  --schedule "$A/outputs/typed_graph_trajectory_${TRAJECTORY_JOB_ID}/typed_readability_schedule.csv" \
  --config "$A/outputs/full_train_shoppers_1033761/config.pkl" \
  --checkpoint "$A/outputs/full_train_shoppers_1033761/ema_model_epoch_8000.pt" \
  --samples 4096 --steps 50 --strength "$GUIDANCE_STRENGTH" --min-support 12 --drift .10 --seed 20261005 \
  --output "$A/outputs/typed_dynamic_graph_guidance_${SLURM_JOB_ID}"
